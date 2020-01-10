#!/usr/bin/env bash
DATASET="shrec_16"
ARCH="meshattentionnet"
DATETIME=`date +%Y-%m-%d_%H-%M-%S`
NAME="${DATASET}_${ARCH}_${DATETIME}"

LOGDIR="logs/${NAME}"
mkdir -p ${LOGDIR}
LOGFILE="${LOGDIR}/${NAME}.log"
exec &> >(tee ${LOGFILE})

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
--niter_decay 100 \
--arch ${ARCH} \