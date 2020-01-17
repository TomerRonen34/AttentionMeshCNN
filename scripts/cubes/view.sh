#!/usr/bin/env bash

FILES=./checkpoints/cubes/*pth

declare -a names=()
for f in $FILES
do
  name=$(echo $f | cut -d'/' -f 4 | cut -d'.' -f 1)
  names+=(${name::-4})
done

for name in ${names[@]}
do
  echo visualizing $name
  python util/mesh_viewer.py \
  --indir C:/3D/Project/checkpoints/cubes/meshes/$name \
  --outdir C:/3D/Project/checkpoints/cubes/pictures/$name
done