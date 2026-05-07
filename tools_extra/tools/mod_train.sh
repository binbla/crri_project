#!/usr/bin/bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export OMP_NUM_THREADS=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --standalone --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch "${@:3}"
