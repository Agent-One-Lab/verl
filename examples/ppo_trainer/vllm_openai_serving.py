from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from fastapi import Request
from collections.abc import AsyncGenerator
from typing import Callable, Optional, Union, cast, Annotated
from pydantic import Field
from vllm.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    RequestResponseMetadata
)
from vllm.sampling_params import BeamSearchParams, SamplingParams
import time
import jinja2
from vllm.logger import init_logger
from vllm.utils import merge_async_iterators
from vllm.outputs import RequestOutput
import asyncio
from vllm.inputs import TokensPrompt
from vllm.entrypoints.openai.serving_engine import CompletionLikeRequest
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.entrypoints.openai.serving_engine import TextTokensPrompt
from vllm.entrypoints.chat_utils import _parse_chat_message_content
from vllm.entrypoints.chat_utils import AsyncMultiModalItemTracker


logger = init_logger(__name__)

def overwrite_vllm_openai_serving_completion(create_completion_with_mm: Callable):
    OpenAIServingCompletion.create_completion = create_completion_with_mm
    return OpenAIServingCompletion

async def _preprocess_completion_with_mm(
    self,
    request: CompletionLikeRequest,
    tokenizer: AnyTokenizer,
    input_or_inputs: Union[str, list[str], list[int], list[list[int]]],
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
    add_special_tokens: bool = True,
) -> tuple[list[TextTokensPrompt], list[TokensPrompt]]:
    
    request_prompts = await self._tokenize_prompt_input_or_inputs_async(
        request,
        tokenizer,
        input_or_inputs,
        truncate_prompt_tokens=truncate_prompt_tokens,
        add_special_tokens=add_special_tokens,
    )


    engine_prompts = [
        TokensPrompt(prompt_token_ids=request_prompt["prompt_token_ids"])
        for request_prompt in request_prompts
    ]
    assert len(engine_prompts) == len(request.messages)
    
    for i, messages in enumerate(request.messages):
        mm_tracker = AsyncMultiModalItemTracker(self.model_config, tokenizer)
        for msg in messages:
            sub_messages = _parse_chat_message_content(
                msg,
                mm_tracker,
                content_format="openai",
            )
        
        mm_data = await mm_tracker.all_mm_data()
        engine_prompts[i]["mm_data"] = mm_data
    
    return request_prompts, engine_prompts

    
async def create_completion_with_mm(
    self,
    request: CompletionRequest,
    raw_request: Optional[Request] = None
) -> Union[AsyncGenerator[str, None], CompletionResponse, ErrorResponse]:
    
    """This is adapted from vllm.entrypoints.openai.serving_completion.OpenAIServingCompletion.create_completion, with _preprocess_completion replaced with _preprocess_completion_with_mm to support multi-modal inputs.

    Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.

    NOTE: Currently we do not support the following feature:
        - suffix (the language models we currently support do not support
        suffix)
    """
    error_check_ret = await self._check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    # If the engine is dead, raise the engine's DEAD_ERROR.
    # This is required for the streaming case, where we return a
    # success status before we actually start generating text :).
    if self.engine_client.errored:
        raise self.engine_client.dead_error

    # Return error for unsupported features.
    if request.suffix is not None:
        return self.create_error_response(
            "suffix is not currently supported")

    request_id = f"cmpl-{self._base_request_id(raw_request)}"
    created_time = int(time.time())

    request_metadata = RequestResponseMetadata(request_id=request_id)
    if raw_request:
        raw_request.state.request_metadata = request_metadata

    try:
        (
            lora_request,
            prompt_adapter_request,
        ) = self._maybe_get_adapters(request)

        tokenizer = await self.engine_client.get_tokenizer(lora_request)

        request_prompts, engine_prompts = await self._preprocess_completion(
            request,
            tokenizer,
            request.prompt,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
            add_special_tokens=request.add_special_tokens,
        )
    except ValueError as e:
        logger.exception("Error in preprocessing prompt inputs")
        return self.create_error_response(str(e))
    except TypeError as e:
        logger.exception("Error in preprocessing prompt inputs")
        return self.create_error_response(str(e))
    except RuntimeError as e:
        logger.exception("Error in preprocessing prompt inputs")
        return self.create_error_response(str(e))
    except jinja2.TemplateError as e:
        logger.exception("Error in preprocessing prompt inputs")
        return self.create_error_response(str(e))

    # Schedule the request and get the result generator.
    generators: list[AsyncGenerator[RequestOutput, None]] = []
    try:
        for i, engine_prompt in enumerate(engine_prompts):
            sampling_params: Union[SamplingParams, BeamSearchParams]
            default_max_tokens = self.max_model_len - len(
                engine_prompt["prompt_token_ids"])
            if request.use_beam_search:
                sampling_params = request.to_beam_search_params(
                    default_max_tokens, self.default_sampling_params)
            else:
                sampling_params = request.to_sampling_params(
                    default_max_tokens,
                    self.model_config.logits_processor_pattern,
                    self.default_sampling_params)

            request_id_item = f"{request_id}-{i}"

            self._log_inputs(request_id_item,
                                request_prompts[i],
                                params=sampling_params,
                                lora_request=lora_request,
                                prompt_adapter_request=prompt_adapter_request)

            trace_headers = (None if raw_request is None else await
                                self._get_trace_headers(raw_request.headers))

            if isinstance(sampling_params, BeamSearchParams):
                generator = self.engine_client.beam_search(
                    prompt=engine_prompt,
                    request_id=request_id,
                    params=sampling_params,
                )
            else:
                generator = self.engine_client.generate(
                    engine_prompt,
                    sampling_params,
                    request_id_item,
                    lora_request=lora_request,
                    prompt_adapter_request=prompt_adapter_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )

            generators.append(generator)
    except ValueError as e:
        # TODO: Use a vllm-specific Validation Error
        return self.create_error_response(str(e))

    result_generator = merge_async_iterators(*generators)

    model_name = self._get_model_name(request.model, lora_request)
    num_prompts = len(engine_prompts)

    # Similar to the OpenAI API, when n != best_of, we do not stream the
    # results. Noting that best_of is only supported in V0. In addition,
    # we do not stream the results when use beam search.
    stream = (request.stream
                and (request.best_of is None or request.n == request.best_of)
                and not request.use_beam_search)

    # Streaming response
    if stream:
        return self.completion_stream_generator(
            request,
            result_generator,
            request_id,
            created_time,
            model_name,
            num_prompts=num_prompts,
            tokenizer=tokenizer,
            request_metadata=request_metadata)

    # Non-streaming response
    final_res_batch: list[Optional[RequestOutput]] = [None] * num_prompts
    try:
        async for i, res in result_generator:
            final_res_batch[i] = res

        for i, final_res in enumerate(final_res_batch):
            assert final_res is not None

            # The output should contain the input text
            # We did not pass it into vLLM engine to avoid being redundant
            # with the inputs token IDs
            if final_res.prompt is None:
                final_res.prompt = request_prompts[i]["prompt"]

        final_res_batch_checked = cast(list[RequestOutput],
                                        final_res_batch)

        response = self.request_output_to_completion_response(
            final_res_batch_checked,
            request,
            request_id,
            created_time,
            model_name,
            tokenizer,
            request_metadata,
        )
    except asyncio.CancelledError:
        return self.create_error_response("Client disconnected")
    except ValueError as e:
        # TODO: Use a vllm-specific Validation Error
        return self.create_error_response(str(e))

    # When user requests streaming but we don't stream, we still need to
    # return a streaming response with a single event.
    if request.stream:
        response_json = response.model_dump_json()

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return fake_stream_generator()

    return response

def overwrite_vllm_openai_serving_completion_with_mm():
    OpenAIServingCompletion._preprocess_completion = _preprocess_completion_with_mm
    OpenAIServingCompletion.create_completion = create_completion_with_mm
    return OpenAIServingCompletion