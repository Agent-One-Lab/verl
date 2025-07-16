
# Run in single node

set -x

export head_node=${nodes[0]}

head_node_ip=$(hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

# export VLLM_ATTENTION_BACKEND=XFORMERS
# export GLOO_SOCKET_IFNAME=ens10f0np0
export HYDRA_FULL_ERROR=1   

# Remove existing Ray cluster
ray stop
rm -rf /tmp/ray/ray_current_cluster

