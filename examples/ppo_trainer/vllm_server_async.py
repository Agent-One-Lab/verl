import asyncio
import logging
from omegaconf import OmegaConf
import sys
import signal
sys.path.append("../../")

config = OmegaConf.create({
    "hybrid_engine": True,
    "model": {
        "path": "Qwen/Qwen2.5-VL-3B-Instruct",
        "external_lib": None,
        "override_config": { },
        "enable_gradient_checkpointing": True,
        "use_remove_padding": False,
        "use_liger": False,
        "trust_remote_code": False,
    },
    "actor": {
        "strategy": "fsdp",  # [fsdp, fsdp2], This is for backward-compatibility
        "ppo_mini_batch_size": 256,
        "ppo_micro_batch_size": None, # will be deprecated, use ppo_micro_batch_size_per_gpu
        "ppo_micro_batch_size_per_gpu": None,
        "use_dynamic_bsz": False,
        "ppo_max_token_len_per_gpu": 16384, # n * ${data.max_prompt_length} + ${data.max_response_length}
        "grad_clip": 1.0,
    # pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
        "clip_ratio": 0.2, # default value if clip_ratio_low and clip_ratio_high are not specified
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "clip_ratio_c": 3.0, # lower bound of the value for Dual-clip PPO from https://arxiv.org/pdf/1912.09729
        "loss_agg_mode": "token-mean", # / "seq-mean-token-sum" / "seq-mean-token-mean"
        "entropy_coeff": 0,
        "use_kl_loss": False, # True for GRPO
        "use_torch_compile": True, # False to disable torch compile
        "kl_loss_coef": 0.001, # for grpo
        "kl_loss_type": "low_var_kl", # for grpo
        "ppo_epochs": 1,
        "shuffle": False,
        "ulysses_sequence_parallel_size": 1, # sp size
    "checkpoint": {
        "contents": ['model', 'optimizer', 'extra'],  # with 'hf_model' you can save whole model as hf format, now only use sharded model checkpoint to save space
        "optim": {
            "lr": 1e-6,
            "lr_warmup_steps": -1, # Prioritized. Negative values mean delegating to lr_warmup_steps_ratio.
            "lr_warmup_steps_ratio": 0.,  # the total steps will be injected during runtime
            "min_lr_ratio": None,   # only useful for warmup with cosine
            "warmup_style": "constant",  # select from constant/cosine
            "total_training_steps": -1,  # must be override by program
            "weight_decay": 0.01,
        },
        "fsdp_config": {
            "wrap_policy": {
                # transformer_layer_cls_to_wrap: None
                "min_num_params": 0,
            },
            "param_offload": False,
            "optimizer_offload": False,
            "offload_policy": False, # only for fsdp2, offload param\grad\optimizer during train
            "reshard_after_forward": True, # only for fsdp2, [True, False, int between 1 and fsdp_size]
            "fsdp_size": -1,
        },
    },
    "ref": {
        "strategy": "fsdp",
        "fsdp_config": {
            "param_offload": False,
            "reshard_after_forward": True, # only for fsdp2, [True, False, int between 1 and fsdp_size]
            "wrap_policy": {
                # transformer_layer_cls_to_wrap: None
                "min_num_params": 0,
            },
            "fsdp_size": -1,
        },
    },
    "log_prob_micro_batch_size": None, # will be deprecated, use log_prob_micro_batch_size_per_gpu
    "log_prob_micro_batch_size_per_gpu": None,
    "log_prob_use_dynamic_bsz": False,
        "log_prob_max_token_len_per_gpu": 16384,
        "ulysses_sequence_parallel_size": 1, # sp size
    },
    "rollout": {
        "name": "vllm",
        "mode": "async", # sync: LLM, async: AsyncLLM
        "chat_scheduler": "examples.ppo_trainer.naive_completion_scheduler.NaiveCompletionScheduler", # async chat scheduler, e.g examples.ppo_trainer.naive_chat_scheduler.NaiveChatCompletionScheduler
        "temperature": 1.0,
        "top_k": -1, # 0 for hf rollout, -1 for vllm rollout
        "top_p": 1,
        "use_fire_sampling": False, # https://arxiv.org/abs/2410.21236
        "prompt_length": 1024,  # not use for opensource
        "response_length": 1024,
        # for vllm rollout
        "dtype": "bfloat16", # should align with FSDP
        "gpu_memory_utilization": 0.5,
        "ignore_eos": False,
        "enforce_eager": True,
        "free_cache_engine": True,
        "load_format": "dummy_dtensor",
        "tensor_model_parallel_size": 1,
        "max_num_batched_tokens": 8192,
        "max_model_len": 8192,
        "max_num_seqs": 1024,
        "log_prob_micro_batch_size": None, # will be deprecated, use log_prob_micro_batch_size_per_gpu
        "log_prob_micro_batch_size_per_gpu": None,
        "log_prob_use_dynamic_bsz": False,
        "log_prob_max_token_len_per_gpu": 16384,
        "disable_log_stats": True,
        "enable_chunked_prefill": True, # may get higher throughput when set to True. When activated, Please increase max_num_batched_tokens or decrease max_model_len.
        # for hf rollout
        "do_sample": True,
        # number of responses (i.e. num sample times)
        "n": 1, # > 1 for grpo
        "engine_kwargs": {
            "swap_space": None, # null means "use the engine default value" (usually 4 GB), setting it to, e.g., 32 means 32 GB
        },
        "val_kwargs": {
      # sampling parameters for validation
            "top_k": -1, # 0 for hf rollout, -1 for vllm rollout
            "top_p": 1.0,
            "temperature": 0,
            "n": 1,
            "do_sample": False, # default eager for validation
        },
        "multi_turn": {
            "enable": False,  # should set rollout.name to sglang_async if True
            "max_turns": None,  # null for no limit (default max_length // 3)
            "tool_config_path": None,  # null for no tool
            "format": "chatml",  # chatml, more formats will be supported in the future
        },
    },
})

roll_tp_size = 1
rollout_dp_size = 1
wg_prefix = "async_llm_server"
rollout_dp_rank = 0

# def start_background_loop(loop: asyncio.AbstractEventLoop):
#     asyncio.set_event_loop(loop)          # bind to this thread
#     loop.run_forever()

# bg_loop = asyncio.new_event_loop()
# bg_thread = threading.Thread(
#     target=start_background_loop, args=(bg_loop,), daemon=True
# )
# bg_thread.start()

async def start_server():
    from vllm_async_server import AsyncvLLMServer

    server = AsyncvLLMServer(
        config=config,
        vllm_dp_size=rollout_dp_size,
        vllm_dp_rank=rollout_dp_rank,
        wg_prefix=wg_prefix,
    )

    try:
        address = await server.get_server_address()
        print(f"Server address: {address}")

        await server.init_engine()
        print("Server initialised")

        # ───── keep running ──────────────────────────────
        # 1) wait for Ctrl-C (SIGINT/SIGTERM handled by asyncio)
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(
                sig, stop.set)

        await stop.wait()       # ← blocks here indefinitely
    finally:
        logging.info("Shutting server down…")

if __name__ == "__main__":
    asyncio.run(start_server(), debug=True)



