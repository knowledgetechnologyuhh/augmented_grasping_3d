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


### In this experiment, we vary the loss weight factors of the retinanet and transformer auxiliary output branches
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
  cd $project_dir/modules/data_fusion/fusionnet/keras_fusionnet/bin/

  # perform 3 rounds
  for i in {1..3}
  do
    echo "ROUND $i \n\n"
    # only fusionnet regression
    python3 train.py --steps=5000 --val-steps=500 --no-snapshots --epochs=10 --batch-size=1 \
     --experiment-tag="ALL" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_weightedloss_exp" --comet-workspace="$COMET_WORKSPACE" \
    fusionnet $project_dir/datasets/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $project_dir/datasets/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="retinanet_regression:1,retinanet_classification:1,transformer_classification:1,fusionnet_regression:1" \
    --coordinate-limits $project_dir/datasets/GeneratedData/FusionMultimodalCSV/virtual/Annotations/joints_min_max.npz \
    vision retinanet $project_dir/datasets/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/train_cropped.csv \
    $project_dir/datasets/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/classmaps.csv \
    --val-annotations $project_dir/datasets/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/val_cropped.csv \
    --image-min-side=360 --image-max-side=660 --no-evaluation language_translation transformer \
    $project_dir/datasets/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $project_dir/datasets/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $project_dir/datasets/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $project_dir/datasets/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt \
    --config $scripts_dir/test_config/transformer_config.ini --no-evaluation

    # all losses
    python3 train.py --steps=5000 --val-steps=500 --no-snapshots --epochs=10 --batch-size=1 \
    --experiment-tag="FUSIONNET_ONLY" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_weightedloss_exp" --comet-workspace="$COMET_WORKSPACE" \
    fusionnet $project_dir/datasets/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $project_dir/datasets/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="fusionnet_regression" --coordinate-limits $project_dir/datasets/GeneratedData/FusionMultimodalCSV/virtual/Annotations/joints_min_max.npz \
    vision retinanet $project_dir/datasets/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/train_cropped.csv \
    $project_dir/datasets/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/classmaps.csv \
    --val-annotations $project_dir/datasets/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/val_cropped.csv \
    --image-min-side=360 --image-max-side=660 --no-evaluation \
    language_translation transformer $project_dir/datasets/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $project_dir/datasets/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $project_dir/datasets/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $project_dir/datasets/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt \
    --config $scripts_dir/test_config/transformer_config.ini --no-evaluation

    # weights experiment
    python3 hyperparameter_optimization.py --steps=5000 --val-steps=500 --no-snapshots --epochs=10 --batch-size=1 \
    --experiment-tag="WMT" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_weightedloss_exp" --comet-workspace="$COMET_WORKSPACE" \
    hyperopt "fusionnet_weights_exp" $project_dir/modules/data_fusion/fusionnet/logs/hyperopt_final_results \
    --optimizer="tpe" --trials-number=3 --max-evals-number=300 --save-interval=1 \
    fusionnet $project_dir/datasets/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $project_dir/datasets/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="retinanet_regression,retinanet_classification,transformer_classification,fusionnet_regression" \
    --coordinate-limits $project_dir/datasets/GeneratedData/FusionMultimodalCSV/virtual/Annotations/joints_min_max.npz \
    vision retinanet $project_dir/datasets/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/train_cropped.csv \
    $project_dir/datasets/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/classmaps.csv \
    --val-annotations $project_dir/datasets/GeneratedData/VisionMultimodalCSV/virtual/Annotations_no_pos/val_cropped.csv \
    --image-min-side=360 --image-max-side=660 --no-evaluation \
    language_translation transformer $project_dir/datasets/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $project_dir/datasets/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $project_dir/datasets/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $project_dir/datasets/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt \
    --config $scripts_dir/test_config/transformer_config.ini --no-evaluation
  done

else
  echo "Activation failed - skipping experiments. Expected \"$virtual_env_name\" but got $(if [[ -z "$VIRTUAL_ENV" ]]; then echo "nothing"; else basename "$VIRTUAL_ENV"; fi) instead"
fi

cd $scripts_dir
