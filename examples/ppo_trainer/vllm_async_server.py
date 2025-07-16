# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import ABC, abstractmethod
import asyncio
from contextlib import asynccontextmanager
import logging
from collections.abc import AsyncGenerator
import os
import socket
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle
import fastapi
import ray
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
import uvicorn
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse, CompletionRequest, CompletionResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor
from vllm.worker.worker_base import WorkerWrapperBase

from verl.utils.fs import copy_to_local
from verl.workers.rollout.async_server import AsyncServerBase
from vllm_openai_serving import overwrite_vllm_openai_serving_completion_with_mm

logger = logging.getLogger(__file__)


def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class AsyncServerBase(ABC):
    """Base class for AsyncServer."""

    def __init__(self):
        self.address = ray._private.services.get_node_ip_address()
        self.port = None
        self.server_ready = asyncio.Event()
        asyncio.create_task(self._start_fastapi_server())

    async def _start_fastapi_server(self):
        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            print("FastAPI startup")
            self.server_ready.set()
            yield

            # There's no way to gracefully restart uvicorn server if port is already in use,
            # so we exit the process directly and let AsyncLLMServerManager restart it.
            print("FastAPI shutdown, maybe address already in use, exit process immediately.")
            os._exit(-1)

        app = fastapi.FastAPI(lifespan=lifespan)
        app.router.add_api_route("/v1/chat/completions", self.chat_completion, methods=["POST"])
        # app.router.add_api_route("/v1/completions", self.completion, methods=["POST"])

        self.port = _get_free_port()
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port, log_level="debug")
        server = uvicorn.Server(config)
        await server.serve()

    async def get_server_address(self) -> Tuple[str, int]:
        """Get FastAPI server address."""
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"

    # @abstractmethod
    # async def chat_completion(self, raw_request: Request):
    #     """OpenAI chat completion API.

    #     API reference: https://platform.openai.com/docs/api-reference/chat/create
    #     """
    #     raise NotImplementedError

    @abstractmethod
    async def completion(self, raw_request: Request):
        """OpenAI completion API.

        API reference: https://platform.openai.com/docs/api-reference/completions/create
        """
        raise NotImplementedError

    @abstractmethod
    async def init_engine(self):
        """Init async LLM engine."""
        raise NotImplementedError

    @abstractmethod
    async def wake_up(self):
        """Wake up engine to load model weights and build kv cache."""
        raise NotImplementedError

    @abstractmethod
    async def sleep(self):
        """Sleep engine to offload model weights and discard kv cache."""
        raise NotImplementedError
    
class AsyncvLLMServer(AsyncServerBase):
    """
    AsyncvLLMServer is a wrapper for AsyncLLM, it uses ExternalRayDistributedExecutor to launch engines
    in hybrid rollout workers, i.e AsyncActorRolloutRefWorker.

    AsyncvLLMServer works as follows:
    1. Start FastAPI server first.
    2. Initialize AsyncLLM with ExternalRayDistributedExecutor.
    3. AsyncLLM spawn EngineCore in subprocess.
    4. EngineCore initialize ExternalRayDistributedExecutor.
    5. ExternalRayDistributedExecutor lookup its corresponding actors by name.
    6. ExternalRayDistributedExecutor init executor: init_worker, init_device, load_model.

    For vLLM AsyncLLM design, see: https://github.com/vllm-project/vllm/pull/9826
    """

    def __init__(self, config: DictConfig, vllm_dp_size: int, vllm_dp_rank: int, wg_prefix: str):
        """
        Args:
            config: DictConfig, actor_rollout_ref config.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
        """
        super().__init__()

        self.config = config
        self.vllm_dp_size = vllm_dp_size
        self.vllm_dp_rank = vllm_dp_rank
        self.wg_prefix = wg_prefix
        self.engine: AsyncLLM = None

    async def init_engine(self):
        """Init vLLM AsyncLLM engine."""
        config = self.config
        model_path = config.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        trust_remote_code = config.model.get("trust_remote_code", False)
        config = config.rollout

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)
        max_model_len = config.max_model_len if config.max_model_len else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        kwargs = dict(
            n=1,
            # logprobs=0,
            max_tokens=config.response_length,
        )
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        print(f"override_generation_config: {kwargs}")

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=True,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            # distributed_executor_backend=ExternalRayDistributedExecutor,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format="auto",
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=self.vllm_dp_rank,
        )

        # init async llm engine
        vllm_config = engine_args.create_engine_config()
        namespace = ray.get_runtime_context().namespace
        vllm_config.instance_id = f"{namespace}:{self.wg_prefix}:{self.vllm_dp_size}:{self.vllm_dp_rank}"
        print(f"vllm_config: {vllm_config}")
        self.engine = AsyncLLM.from_vllm_config(vllm_config)

        # build serving chat
        model_config = self.engine.model_config
        BASE_MODEL_PATHS = [BaseModelPath(name=model_name, model_path=model_path)]
        models = OpenAIServingModels(self.engine, model_config, BASE_MODEL_PATHS)
        self.openai_serving_chat = OpenAIServingChat(
            self.engine,
            model_config,
            models,
            "assistant",
            request_logger=RequestLogger(max_log_len=4096),
            chat_template=None,
            chat_template_content_format="auto",
        )
        OpenAIServingCompletion = overwrite_vllm_openai_serving_completion_with_mm()
        self.serving_completion = OpenAIServingCompletion(
            self.engine,
            model_config,
            models,
            request_logger=RequestLogger(max_log_len=4096),
        )

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        request_json = await raw_request.json()
        request = ChatCompletionRequest(**request_json)
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())
        
    async def completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        request_json = await raw_request.json()
        request = CompletionRequest(**request_json)
        generator = await self.serving_completion.create_completion(request, raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, CompletionResponse)
            return JSONResponse(content=generator.model_dump())
        
    
    # async def chat_completion_generator(self, request: ChatCompletionRequest) -> AsyncGenerator[Tuple[int, str]]:
    #     """Direct chat completion without FastAPI.

    #     Args:
    #         request: ChatCompletionRequest, request object.

    #     Returns:
    #         AsyncGenerator[Tuple[int, str]]: async generator of (status_code, data) pairs.
    #     """
    #     generator = await self.openai_serving_chat.create_chat_completion(request)
    #     if isinstance(generator, ErrorResponse):
    #         data = generator.model_dump_json(exclude_unset=True)
    #         yield generator.code, f"data: {data}\n\n"

    #     if request.stream:
    #         async for chunk in generator:
    #             yield 200, chunk
    #     else:
    #         assert isinstance(generator, ChatCompletionResponse)
    #         data = generator.model_dump_json(exclude_unset=True)
    #         yield 200, f"data: {data}\n\n"

    async def wake_up(self):
        await self.engine.wake_up()

    async def sleep(self):
        # TODO: https://github.com/vllm-project/vllm/issues/17103
        await self.engine.reset_prefix_cache()
        await self.engine.sleep()