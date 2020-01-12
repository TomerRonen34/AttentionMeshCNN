#!/usr/bin/env bash
DATASET="shrec_16"
ARCH="meshattentionnet"
DATETIME=`date +%Y-%m-%d_%H-%M-%S` # "2020-01-10_16-51-32" #
ADD_TO_NAME="_local5"
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
--attn_max_dist 5 \

# --continue_train \
# --epoch_count 201 \
