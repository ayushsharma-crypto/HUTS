#!/bin/sh

input_path=$1
output_path=$2

if [ -d $input_path ]; then
     echo "$input_path is a directory."
else
     echo "Invalid path."
fi


if [ -d $output_path ]; then
     echo "$output_path is a directory."
else
     echo "Invalid path."
fi


for FILE in $input_path*; 
do 
    name="${FILE%.txt}"; 
    name="${name##*/}"; 
    out_file="${output_path}${name}"
    cmd="python -m dirtorch.extract_features --dataset ImageList('${FILE}') --checkpoint Resnet-101-AP-GeM.pt/Resnet-101-AP-GeM.pt  --output ${out_file} --gpu -1 --whitenp 0.25"
    echo ${cmd}
    ${cmd}
done

# ./exec.sh /home/ayushsharma/Documents/College/RRC/IROS2023/seqVLAD_TRAINING_DATA/data_format/test_data/color_files_for_dir/ /home/ayushsharma/Documents/College/RRC/IROS2023/seqVLAD_TRAINING_DATA/data_format/test_data/dir_gds/