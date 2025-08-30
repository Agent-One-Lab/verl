hf_model_path="Qwen/Qwen2.5-3B-Instruct"
local_dir="Local directory for the model weights"
target_dir="Target directory for the converted model weights"
python scripts/model_merger.py --backend fsdp --hf_model_path $hf_model_path --local_dir ${local_dir} --target_dir ${target_dir}