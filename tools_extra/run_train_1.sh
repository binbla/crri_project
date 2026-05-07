#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"
export GPUS=1
run_training() {
    CONFIG='projects/crri_project/config/FRCNN_R101_FPN_DWD.py'
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
    export NCCL_DEBUG=INFO

    export MKL_NUM_THREADS=8
    export PATH=~/.conda/envs/DD/bin:$PATH
    # ./tools/dist_train.sh $CONFIG $GPUS
    ./tools/mod_train.sh $CONFIG $GPUS
}

run_training
