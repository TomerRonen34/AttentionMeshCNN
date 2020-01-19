#!/usr/bin/env bash
DATASET="human_seg"
ARCH="meshunetwithattention"
DATETIME=`date +%Y-%m-%d_%H-%M-%S`
ADD_TO_NAME="_rpr_global"
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
--dataset_mode segmentation \
--ncf 32 64 128 128 \
--ninput_edges 2250 \
--pool_res 1800 1350 600 \
--resblocks 3 \
--batch_size 12 \
--lr 0.001 \
--num_aug 20 \
--slide_verts 0.2 \
--niter 50 \
--niter_decay 50 \
--prioritize_with_attention \
--attn_use_values_as_is \
--double_attention \
--attn_use_positional_encoding \
--attn_max_relative_position 6 \
