# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import base64
import copy
import io
import json
import logging
import os
import re
import traceback
from collections import defaultdict
from typing import Optional, Union, List
import datasets
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
import shutil
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import pandas as pd
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from PIL import Image
import math
logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \\*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}

def convert_parquet_to_json(parquet_file: str, json_file: str):
    df = pd.read_parquet(parquet_file)
    records = df.to_dict(orient='records')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def any_to_pil(img: Union[Image.Image, str, dict, bytes]) -> Image.Image:
    """
    Convert various image formats to PIL Image.
    
    Args:
        img: Image in one of the following formats:
            - PIL Image: returned as-is
            - str: file path to image or base64 string
            - dict: dictionary containing image data (e.g., {"bytes": bytes_data})
            - bytes: raw image bytes
    
    Returns:
        PIL Image object
    """
    if isinstance(img, Image.Image):
        return img
    elif isinstance(img, dict):
        if "bytes" in img:
            img = img["bytes"]
        else:
            raise ValueError("Dictionary must contain 'bytes' key")
    
    if isinstance(img, str):
        # Check if it's a base64 string
        if img.startswith('data:image/'):
            # Handle data URI format
            import base64
            header, encoded = img.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            return Image.open(io.BytesIO(img_bytes))
        elif len(img) > 100 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in img):
            # Likely a base64 string (check for base64 characters and reasonable length)
            try:
                import base64
                img_bytes = base64.b64decode(img)
                return Image.open(io.BytesIO(img_bytes))
            except Exception:
                # If base64 decoding fails, treat as file path
                pass
        # Treat as file path
        return Image.open(img)
    elif isinstance(img, bytes):
        return Image.open(io.BytesIO(img))
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")

def auto_resize_image(img: Image.Image, max_pixels: int = 1536 * 1536, prevent_upscale: bool = True) -> Image.Image:
    w, h = img.size
    area = w * h
    if area <= max_pixels and prevent_upscale:
        return img  # already within budget
    scale = math.sqrt(max_pixels / area) if area > 0 else 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    # Use high-quality downsampling
    return img.resize((new_w, new_h), Image.LANCZOS)


def image_to_data_uri(img: Union[Image.Image, str, dict], fmt=None) -> str:
    if isinstance(img, dict):
        if "bytes" in img:
            img = img["bytes"]

    if isinstance(img, Image.Image):
        # Try to detect format from PIL Image first
        detected_fmt = img.format or fmt or "PNG"
        buf = io.BytesIO()
        img.save(buf, format=detected_fmt)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/{detected_fmt.lower()};base64,{b64}"
    elif isinstance(img, str):
        return img
    elif isinstance(img, bytes):
        # Try to detect format from magic bytes
        detected_fmt = fmt or detect_image_format_from_bytes(img)
        return f"data:image/{detected_fmt.lower()};base64,{base64.b64encode(img).decode('utf-8')}"
    else:
        raise ValueError(f"Invalid image type: {type(img)}")

def detect_image_format_from_bytes(img_bytes: bytes) -> str:
    """Detect image format from bytes using magic numbers"""
    if len(img_bytes) < 4:
        return "PNG"  # Default fallback
    
    # Check magic bytes for common formats
    if img_bytes.startswith(b'\xff\xd8\xff'):
        return "JPEG"
    elif img_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        return "PNG"
    elif img_bytes.startswith(b'GIF87a') or img_bytes.startswith(b'GIF89a'):
        return "GIF"
    elif img_bytes.startswith(b'RIFF') and img_bytes[8:12] == b'WEBP':
        return "WEBP"
    elif img_bytes.startswith(b'BM'):
        return "BMP"
    else:
        return "PNG"  # Default fallback


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.image_patch_size = config.get("image_patch_size", 14)
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.tool_config_path = config.get("tool_config_path", None)
        self.tool_schemas = None
        if self.tool_config_path:
            try:
                from verl.tools.utils.tool_registry import initialize_tools_from_config

                tool_list = initialize_tools_from_config(self.tool_config_path)
                # match ToolAgentLoop behaviour: model_dump to plain dicts
                self.tool_schemas = [
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
                ]
            except Exception as e:
                logger.warning("Failed to initialize tools from %s: %s", self.tool_config_path, e)
                self.tool_schemas = None

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    try:
                        messages = self._build_messages(doc)
                        # pass tool schemas if available so the processor can format prompts
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        raw_prompt = self.processor.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
                        )
                        if image_key in doc and doc[image_key]:
                            images = [
                                process_image(image, image_patch_size=self.image_patch_size) for image in doc[image_key]
                            ]
                        else:
                            images = None

                        if video_key in doc and doc[video_key]:
                            videos, video_metadata = zip(
                                *[
                                    process_video(
                                        video, image_patch_size=self.image_patch_size, return_video_metadata=True
                                    )
                                    for video in doc[video_key]
                                ],
                                strict=True,
                            )
                            videos = list(videos)
                            video_metadata = list(video_metadata)
                            videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}
                        else:
                            videos = None
                            videos_kwargs = {}

                        return len(
                            processor(text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs)[
                                "input_ids"
                            ][0]
                        )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            else:

                def doc2len(doc) -> int:
                    try:
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        return len(
                            tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True, **apply_kwargs)
                        )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [process_image(image, image_patch_size=self.image_patch_size) for image in row_dict_images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            videos_kwargs = {}
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos, video_metadata = zip(
                    *[
                        process_video(video, image_patch_size=self.image_patch_size, return_video_metadata=True)
                        for video in row_dict_videos
                    ],
                    strict=True,
                )
                videos = list(videos)
                video_metadata = list(video_metadata)
                videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [
                    (video.numpy(), metadata) for video, metadata in zip(videos, video_metadata, strict=True)
                ]

            model_inputs = self.processor(
                text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs, return_tensors="pt"
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()


class RLHFAgentDataset(Dataset):
    def __init__(self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        processor = None, # Compatible with verl
        config = None, # Compatible with verl,
        max_samples: int = -1,
    ):
        # print(f'data_files: {data_files}')
        self.tokenizer = tokenizer
        self.data_files = copy.deepcopy(data_files)
        self.data = []
        self.sources = []
        self.truncation = "error"
        self.max_samples = max_samples

        if isinstance(self.data_files, str):
            self.data_files = [self.data_files]
        elif isinstance(self.data_files, list):
            self.data_files = [f for f in self.data_files]
        else:
            raise ValueError(f"Unsupported data_files type: {type(self.data_files)}")
        # for i, data_file in enumerate(self.data_files):
        #     self.data.extend(json.load(open(data_file)))
        #     file_name = os.path.basename(data_file)
        #     self.sources.extend([file_name] * len(json.load(open(data_file))))
        self._download_data()
        self._read_data()

    def _download_data(self):
        for data_file in self.data_files:
            if not os.path.exists(data_file):
                # Try to find the data file in HF repo
                filename = os.path.basename(data_file)
                try:
                    cached_path = hf_hub_download(
                        repo_id="Agent-One/AgentFly-Train",
                        filename=filename,
                        repo_type="dataset",
                    )
                except EntryNotFoundError:
                    raise ValueError(f"Data file {data_file} not found in local directory nor in HF repo Agent-One/AgentFly-Train.")
                folder = os.path.dirname(data_file)
                os.makedirs(folder, exist_ok=True)
                shutil.copy(cached_path, data_file)
                print(f"Downloaded {filename} to {data_file}")

    def _read_data(self):
        # self._convert_parquet_to_json(self.data_files)
        # json_files = [f for f in self.data_files if f.endswith('.json')]

        # if json_files:
        #     for json_file in json_files:
        #         with open(json_file, 'r', encoding='utf-8') as f:
        #             json_data = json.load(f)
        #             self.data.extend(json_data)
        #             file_name = os.path.basename(json_file)
        #             self.sources.extend([file_name] * len(json_data))
        for data_file in self.data_files:
            if data_file.endswith('.parquet'):
                df = pd.read_parquet(data_file)
                self.data.extend(df.to_dict(orient='records'))
                self.sources.extend([os.path.basename(data_file)] * len(df))
            elif data_file.endswith('.json'):
                with open(data_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    self.data.extend(json_data)
                    self.sources.extend([os.path.basename(data_file)] * len(json_data))
            else:
                raise ValueError(f"Unsupported file type: {data_file}")
        print(f"dataset len: {len(self.data)}")

        if self.max_samples > 0 and self.max_samples < len(self.data):
            sampled_ids = np.random.choice(len(self.data), self.max_samples, replace=False)
            self.data = [self.data[i] for i in sampled_ids]
            self.sources = [self.sources[i] for i in sampled_ids]
        

    def _convert_parquet_to_json(self, files):
        json_files = []
        for file in files:
            if file.endswith('.parquet'):
                json_path = file.replace('.parquet', '.converted.json')
                if not os.path.exists(json_path):
                    convert_parquet_to_json(file, json_path)
                json_files.append(json_path)
            else:
                json_files.append(file)
        return json_files

    def __len__(self):
        return len(self.data)

    def _process_image(self, image):
        """
        Process image. First limits the maximum pixels and scale it. Then convert to data URI (base64)
        """
        image = any_to_pil(image)
        image = auto_resize_image(image)
        image = image_to_data_uri(image)
        return image
    
    def _build_messages(self, row_dict):
        question_keys = ['question', 'problem', 'instruction']
        for key in question_keys:
            question = None
            if key in row_dict:
                question = row_dict[key]
                break
        if question is None:
            raise ValueError(f"question not found in row_dict: {row_dict}")
        
        if "image" in row_dict:
            from verl.utils.dataset.vision_utils import process_image
            # image = process_image(row_dict["image"])
            image = row_dict["image"]
            image = self._process_image(image)
            # convert PIL Image to base64
            # buffer = io.BytesIO()
            # image.save(buffer, format="PNG")
            # image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
        else:
            image = None
        
        single_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question}
                ]
            }
        ]
        if image is not None:
            # OpenAI chat completion API only supports image_url
            single_messages[0]["content"].append({"type": "image_url", "image_url": {"url": image}})
        messages = {
            "messages": single_messages,
            "question": question,
        }

        other_info = {}
        for k, v in row_dict.items():
            if k not in ['question']:
                other_info[k] = v

        messages.update(other_info)

        return messages, question


    
    def __getitem__(self, item):
        row_dict = self.data[item]
        
        
        messages, question = self._build_messages(row_dict)
        row_dict["messages"] = messages
        row_dict["data_source"] = self.sources[item]
        row_dict["question"] = question
        # May be for compatibility with the original dataset
        # And we don't actually need this
        # inputs = self.tokenizer(question, return_tensors='pt')
        # row_dict["input_ids"] = inputs.input_ids
        # row_dict["attention_mask"] = inputs.attention_mask
        row_dict["input_ids"] = torch.tensor([0])
        row_dict["attention_mask"] = torch.tensor([1])
        row_dict["position_ids"] = torch.tensor([0])
        
        return row_dict