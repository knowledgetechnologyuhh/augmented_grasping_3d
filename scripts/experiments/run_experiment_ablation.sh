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


### In this experiment, we ablate the retinanet and transformer auxiliary output branches
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
  #env vars
  source $project_dir/scripts/env_vars.sh
  
  # export gpu
  export CUDA_VISIBLE_DEVICES=${2:-0}

  # setup all the packages
  cd $project_dir/modules/data_fusion/fusionnet/keras_fusionnet/bin/

  # perform 3 rounds
  for i in {1..3}
  do
    echo "ROUND $i \n\n"
    # the full network
    python3 train.py --steps=5000 --val-steps=500 --no-snapshots --epochs=10 --batch-size=1 \
    --experiment-tag="FULL" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_exp" --comet-workspace="$COMET_WORKSPACE" \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="retinanet_regression,retinanet_classification,transformer_classification,fusionnet_regression" \
    --coordinate-limits $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/joints_min_max.npz \
    --node-number 4 \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/train_cropped.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/val_cropped.csv \
    --image-min-side=360 --image-max-side=660 --no-evaluation \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt \
    --config $scripts_dir/test_config/transformer_config.ini --no-evaluation

    # no transformer output, no transformer validation
    python3 train.py --steps=5000 --val-steps=500 --no-snapshots --epochs=10 --batch-size=1 \
    --experiment-tag="NO_TRANS" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_exp" --comet-workspace="$COMET_WORKSPACE" \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="retinanet_regression,retinanet_classification,fusionnet_regression" \
    --coordinate-limits $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/joints_min_max.npz \
    --node-number 4 \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/train_cropped.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/val_cropped.csv \
    --image-min-side=360 --image-max-side=660 --no-evaluation \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv --val-golden-set \
    $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt \
    --config $scripts_dir/test_config/transformer_config.ini --no-evaluation

    # no transformer output, no transformer validation, no retinanet classification, no retinanet validation
    python3 train.py --steps=5000 --val-steps=500 --no-snapshots --epochs=10 --batch-size=1 \
    --experiment-tag="NO_TRANS_NO_RET_CLAS" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_exp" --comet-workspace="$COMET_WORKSPACE" \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="retinanet_regression,fusionnet_regression" \
    --coordinate-limits $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/joints_min_max.npz \
    --node-number 4 \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/train_cropped.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/val_cropped.csv \
    --image-min-side=360 --image-max-side=660 --no-evaluation \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt \
    --config $scripts_dir/test_config/transformer_config.ini --no-evaluation

    # no transformer output, no transformer validation, no retinanet regression, no retinanet validation
    python3 train.py --steps=5000 --val-steps=500 --no-snapshots --epochs=10 --batch-size=1 \
    --experiment-tag="NO_TRANS_NO_RET_REG" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_exp" --comet-workspace="$COMET_WORKSPACE" \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="retinanet_classification,fusionnet_regression" \
    --coordinate-limits $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/joints_min_max.npz \
    --node-number 4 \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/train_cropped.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/val_cropped.csv \
    --image-min-side=360 --image-max-side=660 --no-evaluation \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt \
    --config $scripts_dir/test_config/transformer_config.ini --no-evaluation

    # no retinanet classification, no retinanet validation
    python3 train.py --steps=5000 --val-steps=500 --no-snapshots --epochs=10 --batch-size=1 \
    --experiment-tag="NO_RET_CLAS" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_exp" --comet-workspace="$COMET_WORKSPACE" \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="retinanet_regression,transformer_classification,fusionnet_regression" \
    --coordinate-limits $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/joints_min_max.npz \
    --node-number 4 \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/train_cropped.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/val_cropped.csv \
    --image-min-side=360 --image-max-side=660 --no-evaluation \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt \
    --config $scripts_dir/test_config/transformer_config.ini --no-evaluation

    # no retinanet regression, no retinanet validation
    python3 train.py --steps=5000 --val-steps=500 --no-snapshots --epochs=10 --batch-size=1 \
    --experiment-tag="NO_RET_REG" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_exp" --comet-workspace="$COMET_WORKSPACE" \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="retinanet_classification,transformer_classification,fusionnet_regression" \
    --coordinate-limits $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/joints_min_max.npz \
    --node-number 4 \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/train_cropped.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/val_cropped.csv \
    --image-min-side=360 --image-max-side=660 --no-evaluation \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt \
    --config $scripts_dir/test_config/transformer_config.ini --no-evaluation

    # no retinanet regression, no retinanet classification, no retinanet validation
    python3 train.py --steps=5000 --val-steps=500 --no-snapshots --epochs=10 --batch-size=1 \
    --experiment-tag="NO_RET_REG_NO_RET_CLAS" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_exp" --comet-workspace="$COMET_WORKSPACE" \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="transformer_classification,fusionnet_regression" \
    --coordinate-limits $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/joints_min_max.npz \
    --node-number 4 \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/train_cropped.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/val_cropped.csv \
    --image-min-side=360 --image-max-side=660 --no-evaluation \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt \
    --config $scripts_dir/test_config/transformer_config.ini --no-evaluation

    # no transformer output, no transformer validation, no retinanet classification, no retinanet regression, no retinanet validation
    python3 train.py --steps=5000 --val-steps=500 --no-snapshots --epochs=10 --batch-size=1 \
    --experiment-tag="NO_TRANS_NO_RET_CLAS_NO_RET_REG" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_exp" --comet-workspace="$COMET_WORKSPACE" \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="fusionnet_regression" \
    --coordinate-limits $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/joints_min_max.npz \
    --node-number 4 \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/train_cropped.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/val_cropped.csv \
    --image-min-side=360 --image-max-side=660 --no-evaluation \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt \
    --config $scripts_dir/test_config/transformer_config.ini --no-evaluation
  done

else
  echo "Activation failed - skipping experiments. Expected \"$virtual_env_name\" but got $(if [[ -z "$VIRTUAL_ENV" ]]; then echo "nothing"; else basename "$VIRTUAL_ENV"; fi) instead"
fi

cd $scripts_dir
