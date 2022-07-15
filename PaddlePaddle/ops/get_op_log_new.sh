#!/bin/bash

fullname=$(basename -- "${1}")
#extension="${fullname##*.}"
filename="${fullname%.*}"

echo "fullname=${fullname}"
echo "filename=${filename}"
#echo "extension=${extension}"

cat ${1}| grep -a "API kernel key" | cut -d " " -f5 > "${filename}_temp.log"
sort -u "${filename}_temp.log" > "${filename}_$(date +'%Y-%m-%d').txt"
rm -rf "${filename}_temp.log"

