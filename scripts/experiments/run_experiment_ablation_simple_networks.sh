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


### In this experiment, we simplify the networks: reduce the retinanet to a cnn; autoencode english-english (no RCL)
# get dir
scripts_dir=`pwd`
project_dir="$(dirname "$(dirname "$scripts_dir")")"
echo "The project directory: " $project_dir

cd $project_dir
echo "Activating virtualenv"
source $virtual_env_name"/bin/activate"

# inside the virtual environment, run all setups
if [[ $(basename "$VIRTUAL_ENV") == $virtual_env_name ]]; then
  echo "Running experiments"

  # export gpu
  export CUDA_VISIBLE_DEVICES=${2:-0}

  # setup all the packages
  cd $project_dir/modules/data_fusion/fusionnet/keras_fusionnet/bin/

  # perform 3 rounds
  for i in {1..3}
  do
    echo "ROUND $i \n\n"
    # no retinanet loss and using cnn instead
    python3 train.py --steps=5000 --val-steps=500 --no-snapshots --epochs=10 --batch-size=1 \
    --experiment-tag "CNN_NO_RET_LOSS" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_simplenet_exp" --comet-workspace="$COMET_WORKSPACE" \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="transformer_classification:1,fusionnet_regression:1" \
    --coordinate-limits $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/joints_min_max.npz \
    vision simple_cnn $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/train_cropped.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/val_cropped.csv \
    --image-min-side=360 --image-max-side=660 --no-evaluation \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt \
    --config $scripts_dir/test_config/transformer_config.ini --no-evaluation

    # all losses with english as source and target
    python3 train.py --steps=5000 --val-steps=500 --no-snapshots --epochs=10 --batch-size=1 \
    --experiment-tag "ENGLISH_SOURCE_TARGET" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_simplenet_exp" --comet-workspace="$COMET_WORKSPACE" \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="retinanet_regression:1,retinanet_classification:1,transformer_classification:1,fusionnet_regression:1" \
    --coordinate-limits $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/joints_min_max.npz \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/train_cropped.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/val_cropped.csv \
    --image-min-side=360 --image-max-side=660 --no-evaluation \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations_english_only/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations_english_only/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations_english_only/val.csv \
    --config $scripts_dir/test_config/transformer_config.ini --no-evaluation

    # no fusionnet loss with english as source and target
    python3 train.py --steps=5000 --val-steps=500 --no-snapshots --epochs=10 --batch-size=1 \
    --experiment-tag "ENGLISH_SOURCE_TARGET_NO_TRAN_LOSS" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_simplenet_exp" --comet-workspace="$COMET_WORKSPACE" \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="retinanet_regression:1,retinanet_classification:1,fusionnet_regression:1" \
    --coordinate-limits $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/joints_min_max.npz \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/train_cropped.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/val_cropped.csv \
    --image-min-side=360 --image-max-side=660 --no-evaluation \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations_english_only/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations_english_only/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations_english_only/val.csv \
    --config $scripts_dir/test_config/transformer_config.ini --no-evaluation
  done

else
  echo "Activation failed - skipping experiments. Expected \"$virtual_env_name\" but got $(if [[ -z "$VIRTUAL_ENV" ]]; then echo "nothing"; else basename "$VIRTUAL_ENV"; fi) instead"
fi

cd $scripts_dir
