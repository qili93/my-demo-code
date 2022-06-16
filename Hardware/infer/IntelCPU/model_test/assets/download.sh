#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

echo "---------------Prepare target dirs-----------------"
target_dir=$cur_dir/models
if [ ! -d "$target_dir" ]; then
  mkdir -p "$target_dir"
fi
cd $target_dir

echo "---------------Download Lite Models-----------------"
# wget 


echo "---------------Unzip Lite Models-----------------"
tar zxf ruliu_lite_models.tar.gz

cd -