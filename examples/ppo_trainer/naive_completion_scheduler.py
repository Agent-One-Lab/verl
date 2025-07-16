# An variant version of NaiveChatCompletionScheduler for completion scheduler
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
import asyncio
from typing import Any, Dict, List

import torch
from openai.types.completion import Completion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.workers.rollout.async_server import CompletionScheduler


class NaiveCompletionScheduler(CompletionScheduler):
    """
    A very naive implementation of CompletionScheduler for demo purpose,
    only do single-turn completion.
    """

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        # print(f"[NaiveCompletionScheduler] generate_sequences batch: {batch[0]}")

        kwargs = dict(
            n=self.config.n,
            max_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        # print(f"[NaiveChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        async def callback(completions: Completion, info: Dict[str, Any], exception: Exception):
            # Don't do anything for completion
            batch_completions, batch_index, batch_size = info["batch_completions"], info["batch_index"], info["batch_size"]
            responses = []
            if completions:
                for choice in completions.choices:
                    batch_completions[batch_index] = choice.text
                    responses.append(choice.text)
            else:
                responses = [""] * batch_size

            batch_completions[batch_index] = responses

        # TODO: we may need to control max concurrent requests here, or it will harm prefix cache hit rate.
        tasks = []
        batch_completions = [None] * len(batch)
        for batch_index, prompt in enumerate(batch.non_tensor_batch["raw_prompt"]):
            # prompt is a string
            tasks.append(
                asyncio.create_task(
                    self.submit_completions(
                        callback=callback,
                        callback_additional_info={
                            "batch_completions": batch_completions,
                            "batch_index": batch_index,
                            "batch_size": len(batch_completions),
                        },
                        model=self.model_name,
                        prompt=prompt,
                        **kwargs,
                    )
                )
            )
        await asyncio.gather(*tasks)

        return self._postprocess(batch, batch_completions, n=kwargs["n"])

    def _postprocess(self, batch: DataProto, batch_completions: List[List[str]], n: int) -> DataProto:
        # flatten batch_completions if n > 1
        responses = [completion for completions in batch_completions for completion in completions]
        batch = TensorDict(
            {
                "responses": responses,
            },
            batch_size=len(responses),
        )

        return DataProto(batch=batch)
