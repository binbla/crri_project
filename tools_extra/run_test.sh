#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"

run_test() {
    GPUS=1
    CONFIG='projects/chr_project/config/FRCNN_R101_FPN_DWD.py'
    CHECKPOINT="Runtime/FRCNN_R101_FPN_DWD/iter_30000.pth"
    work_dir="Runtime/FRCNN_R101_FPN_DWD/"
    show_dir="Runtime/FRCNN_R101_FPN_DWD/show/"
    # export PATH=/home/user/.conda/envs/DD/bin:$PATH
    export PORT=29501
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
    export NCCL_SOCKET_IFNAME=lo
    export OMP_NUM_THREADS=1

    # ./tools/dist_test.sh $CONFIG $CHECKPOINT $GPUS #--show-dir ${show_dir}
    ./tools/mod_test.sh $CONFIG  $GPUS $CHECKPOINT  --work-dir ${work_dir} #--show-dir ${show_dir}
}

run_test
