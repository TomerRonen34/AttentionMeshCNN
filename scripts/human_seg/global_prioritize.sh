#!/usr/bin/env bash
DATASET="human_seg"
ARCH="meshunetwithattention"
DATETIME=`date +%Y-%m-%d_%H-%M-%S` # "2020-01-10_16-51-01" #
ADD_TO_NAME="_global_prioritize"
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
--ncf 32 64 128 256 \
--ninput_edges 2250 \
--pool_res 1800 1350 600 \
--resblocks 3 \
--batch_size 12 \
--lr 0.001 \
--num_aug 20 \
--slide_verts 0.2 \
--prioritize_with_attention \
--niter 50 \
--niter_decay 10 \

#--continue_train \
#--epoch_count 201 \
