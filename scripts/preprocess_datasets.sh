#!/bin/bash

cd ${BASH_SOURCE[0]}/..

for x in `find data/scene_datasets/gibson/ -name '*glb'`; do
    ./simulator/python/bps_sim/preprocess $x `dirname $x`/`basename $x .glb`.bps right backward up data/scene_datasets/gibson --texture-dump
done

mkdir textures
mv data/scene_datasets/gibson/*jpg textures/

[ ! -d "data/scene_datasets/mp3d" ] && exit

for x in `find data/scene_datasets/mp3d/ -name '*glb'`; do
    dir="`dirname $x`"

    ./simulator/python/bps_sim/preprocess $x $dir/`basename $x .glb`.bps right backward up $dir
done
