#!/usr/bin/env bash
DATASET="cubes"
ARCH="meshattentionnet"
DATETIME=`date +%Y-%m-%d_%H-%M-%S`
ADD_TO_NAME="_rpr_local5"
NAME="${DATASET}_${ARCH}${ADD_TO_NAME}_${DATETIME}"

LOGDIR="checkpoints/${NAME}"
mkdir -p ${LOGDIR}
LOGFILE="${LOGDIR}/bash_log.log"
exec &> >(tee -a ${LOGFILE})

## run the training
python -u train.py \
--dataroot datasets/${DATASET} \
--name ${NAME} \
--arch ${ARCH} \
--ncf 64 128 256 256 \
--pool_res 600 450 300 180 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter 50 \
--niter_decay 50 \
--prioritize_with_attention \
--attn_use_values_as_is \
--double_attention \
--attn_max_dist 5 \
--attn_use_positional_encoding \
--attn_max_relative_position 5 \

# --continue_train \
# --which_epoch 9 \
# --epoch_count 10 \

