#!/usr/bin/env bash
DATASET="shrec_10_split_0"
ARCH="meshattentionnet"
DATETIME="2020-01-11_19-48-33" # `date +%Y-%m-%d_%H-%M-%S`
ADD_TO_NAME="_local3_prioritize"
NAME="${DATASET}_${ARCH}${ADD_TO_NAME}_${DATETIME}"

LOGDIR="checkpoints/${NAME}"
mkdir -p ${LOGDIR}
LOGFILE="${LOGDIR}/bash_log.log"
exec &> >(tee -a ${LOGFILE})

## run the training
python -u train.py \
--dataroot datasets/${DATASET} \
--name ${NAME} \
--ncf 64 128 256 256 \
--pool_res 600 450 300 180 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 200 \
--arch ${ARCH} \
--attn_max_dist 3 \
--prioritize_with_attention \
 --continue_train \
 --which_epoch 165 \
 --epoch_count 166 \
