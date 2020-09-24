#!/bin/bash

# arg1: (Default:venv3) name of the virtual environment (will be stored in the main directory of this project). Default is stored in the file ./venv . if problems occur, change the name in the file
# arg2: (Default:0) the number of the gpu to use. This script does not support multiple gpus

# get the virtual environment name from the first argument to the script. if none is given, use the last virtual environment.
# change the first line in the ../venv file to match the default virtual environment name if needed
virtual_env_name=${1:-$(<.././venv)}
rm .././venv
echo "$virtual_env_name"
echo "$virtual_env_name">>.././venv
echo "Requested virtual environment: $virtual_env_name"


### In this experiment, we fine-tune the retinanet with the best hyperparams
# get dir
scripts_dir=`pwd`
project_dir="$(dirname "$(dirname "$scripts_dir")")"
echo "The project directory: " $project_dir

cd $project_dir
echo "Activating virtualenv"
source $virtual_env_name"/bin/activate"

# inside the virtual environment, run all setups
if [[ $(basename "$VIRTUAL_ENV") == "$virtual_env_name" ]]; then
  echo "Running experiments"

  # export gpu
  export CUDA_VISIBLE_DEVICES=${2:-0}

  # setup all the packages
  cd $project_dir/modules/vision_object_detection/retinanet/keras_retinanet/bin/

  # perform 3 rounds
  for i in {1..3}
  do
    echo "ROUND $i \n\n"
    # with location bias results
    python3 train.py --steps=100000 --val-steps=31108 --batch-size=1 --backbone='resnet50' --no-snapshot --weighted-average  --image-min-side=540 --image-max-side=990 \
    --experiment-tag="RETINANET" --comet-api-key="$COMET_API_KEY" --comet-project-name="final_models" --comet-workspace="$COMET_WORKSPACE" \
    csvloc $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/train_cropped.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/val_cropped.csv
  done
  
else
  echo "Activation failed - skipping experiments. Expected \"$virtual_env_name\" but got $(if [[ -z "$VIRTUAL_ENV" ]]; then echo "nothing"; else basename "$VIRTUAL_ENV"; fi) instead"
fi

cd $scripts_dir
