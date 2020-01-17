#!/usr/bin/env bash

DATASET="cubes"
FILES=./checkpoints/cubes/*pth

declare -a names=()
for f in $FILES
do
  name=$(echo $f | cut -d'/' -f 4 | cut -d'.' -f 1)
  names+=(${name::-4})
done

echo ${names[@]}

for name in ${names[@]}
do
  IFS='_';  read -ra splitted <<< $name;  IFS=' '
  arch=${splitted[0]}

  args="--dataroot datasets/${DATASET} --name cubes --norm group --resblocks 1 --flip_edges 0.2 --slide_verts 0.2 --num_aug 20 --niter_decay 200 --gpu_ids -1 --which_epoch $name --arch $arch --export_folder meshes/$name"

  if echo $name | grep -q "more_params"; then
    args+=" --ncf 64 128 256 256 256 --pool_res 600 450 300 180 150"
  else
    args+=" --ncf 64 128 256 256 --pool_res 600 450 300 180"
  fi

  if echo $name | grep -q "prioritize"; then
    args+=" --prioritize_with_attention"
  fi

  if echo $name | grep -q "local3"; then
    args+=" --attn_max_dist 3"
  fi

  if echo $name | grep -q "local5"; then
    args+=" --attn_max_dist 5"
  fi

  python test.py $args
done