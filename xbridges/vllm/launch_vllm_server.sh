#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

### Cluster Network Setting - M1
#export OMPI_MCA_coll_hcoll_enable=0 \
#CUDA_DEVICE_ORDER=PCI_BUS_ID \
#NCCL_SOCKET_IFNAME=eth0 \
#UCX_TLS=rc \
#UCX_NET_DEVICES=mlx5_ib0:1 \
#NCCL_DEBUG=WARN \
#NCCL_TOPO_FILE=/opt/microsoft/ndv5-topo.xml \
#NCCL_IB_PCI_RELAXED_ORDERING=1 \
#NCCL_IB_QPS_PER_CONNECTION=4 \
#NCCL_IGNORE_CPU_AFFINITY=1 \
#NCCL_P2P_NET_CHUNKSIZE=$((512 * 1024)) \
#NCCL_PXN_DISABLE=1 \
#NCCL_MIN_NCHANNELS=32 \
#SHARP_SMX_UCX_INTERFACE=mlx5_ib0:1 \
#SHARP_COLL_ENABLE_SAT=1 \
#SHARP_COLL_LOG_LEVEL=3 \
#SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1 \
#NCCL_COLLNET_ENABLE=1

### Cluster Network Setting - M2
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1

### Node-local compile caches

export VLLM_CACHE_ROOT=/tmp/${USER}/vllm-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/${USER}/torchinductor-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/${USER}/triton-${SLURM_JOB_ID}

# Get the list of allocated nodes
nodes_array=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
echo "nodes_array: ${nodes_array[@]}"

head_name=${nodes_array[0]}
ray_port=6379
vllm_port=6380
head_ip=$(srun --nodes=1 --ntasks=1 -w ${head_name} hostname -I | awk '{print $1}')
ray_head=${head_ip}:${ray_port}
echo "head_ip: ${head_ip}"
echo "ray_head: ${ray_head}"
echo "head_name: ${head_name}"
echo "vllm_head: ${head_name}:${vllm_port}"

# ray stop at all nodes
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} --ntasks-per-node=1 ray stop
sleep 10

# Remove existing Ray cluster
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

# Initialize node-local compile caches
srun \
    --nodes=${SLURM_NNODES} \
    --ntasks=${SLURM_NNODES} \
    --ntasks-per-node=1 \
    bash -c '
        rm -rf "$VLLM_CACHE_ROOT" \
               "$TORCHINDUCTOR_CACHE_DIR" \
               "$TRITON_CACHE_DIR"

        mkdir -p "$VLLM_CACHE_ROOT" \
                 "$TORCHINDUCTOR_CACHE_DIR" \
                 "$TRITON_CACHE_DIR"
    '


echo "Starting HEAD at ${head_name}"
srun --nodes=1 --ntasks=1 -w ${head_name} --export=ALL \
  ray start \
    --head \
    --node-ip-address ${head_ip} \
    --port ${ray_port} \
    --num-cpus ${SLURM_CPUS_PER_TASK} \
    --num-gpus ${SLURM_GPUS_PER_NODE} \
    --block &

until nc -z ${head_ip} ${ray_port}; do
  echo "waiting for Ray head port..."
  sleep 5
done

until srun --overlap --nodes=1 --ntasks=1 -w ${head_name} ray status --address ${ray_head} >/dev/null 2>&1; do
  echo "waiting for Ray cluster ready..."
  sleep 5
done

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "Starting WORKER $i at ${node_i}"
  srun --nodes=1 --ntasks=1 -w ${node_i} --export=ALL \
    ray start \
      --address ${ray_head} \
      --num-cpus ${SLURM_CPUS_PER_TASK} \
      --num-gpus ${SLURM_GPUS_PER_NODE} \
      --block &
  sleep 5
done

# launch vllm server
srun --overlap --nodes=1 --ntasks=1 -w ${head_name} --export=ALL \
  vllm serve \
    --distributed-executor-backend ray \
    --model /mnt/weka/shrd/k2m/suqi.sun/bbq_image/k2mova-36b-mid4_v2/checkpoint_0010000 \
    --model-impl vllm \
    --tensor-parallel-size $((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE)) \
    --dtype float32 \
    --trust-remote-code \
    --host ${head_name} \
    --port ${vllm_port}