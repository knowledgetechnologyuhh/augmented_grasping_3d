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

### DEPRECATED: In this experiment, we ablate the retinanet and transformer auxiliary output branches (with the target as the x,y,z and joint angles as block classes)
# get dir
dir=`dirname "$BASH_SOURCE"`
scripts_dir=`pwd`
project_dir="$(dirname "$scripts_dir")"
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
  # this is a test script
  python3 train.py --steps=10 --val-steps=5 --batch-size=1 --epochs=1 --experiment-tag="SANITY_CHECK_TEST" \
	--comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_pos_exp" --comet-workspace="$COMET_WORKSPACE" \
	--no-snapshots \
	fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
	--val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
	--output-filter="retinanet_regression,retinanet_classification,transformer_classification,fusionnet_regression" \
	vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/train.csv \
	$DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/classmaps.csv \
	--val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/val.csv \
	language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
	--vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
	--val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
	--val-golden-set $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt
  
  # perform 3 rounds
  for i in {1..3}
  do
    echo "ROUND $i \n\n"
    # the full network
    python3 train.py --steps=5000 --val-steps=500 --epochs=10 --batch-size=1 --experiment-tag="FULL" \
    --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_pos_exp" --comet-workspace="$COMET_WORKSPACE" \
    --no-snapshots \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="retinanet_regression,retinanet_classification,transformer_classification,fusionnet_regression" \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/train.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/val.csv \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt

    # no transformer output, no transformer validation
    python3 train.py --steps=5000 --val-steps=500 --epochs=10 --batch-size=1 --experiment-tag="NO_TRANS" \
    --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_pos_exp" --comet-workspace="$COMET_WORKSPACE" \
    --no-snapshots \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="retinanet_regression,retinanet_classification,fusionnet_regression" \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/train.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/val.csv \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt --no-evaluation

    # no transformer output, no transformer validation, no retinanet classification, no retinanet validation
    python3 train.py --steps=5000 --val-steps=500 --epochs=10 --batch-size=1 --experiment-tag="NO_TRANS_NO_RET_CLAS" \
    --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_pos_exp" --comet-workspace="$COMET_WORKSPACE" \
    --no-snapshots \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="retinanet_regression,fusionnet_regression" \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/train.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/val.csv --no-evaluation \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt --no-evaluation

    # no transformer output, no transformer validation, no retinanet regression, no retinanet validation
    python3 train.py --steps=5000 --val-steps=500 --epochs=10 --batch-size=1 --experiment-tag="NO_TRANS_NO_RET_REG" \
    --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_pos_exp" --comet-workspace="$COMET_WORKSPACE" \
    --no-snapshots \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="retinanet_classification,fusionnet_regression" \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/train.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/val.csv --no-evaluation \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt --no-evaluation

    # no retinanet classification, no retinanet validation
    python3 train.py --steps=5000 --val-steps=500 --epochs=10 --batch-size=1 --experiment-tag="NO_RET_CLAS" \
    --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_pos_exp" --comet-workspace="$COMET_WORKSPACE" \
    --no-snapshots \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="transformer_classification,retinanet_regression,fusionnet_regression" \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/train.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/val.csv --no-evaluation \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt

    # no retinanet regression, no retinanet validation
    python3 train.py --steps=5000 --val-steps=500 --epochs=10 --batch-size=1 --experiment-tag="NO_RET_REG" \
    --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_pos_exp" --comet-workspace="$COMET_WORKSPACE" \
    --no-snapshots \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="transformer_classification,retinanet_classification,fusionnet_regression" \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/train.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/val.csv --no-evaluation \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt

    # no retinanet regression, no retinanet classification, no retinanet validation
    python3 train.py --steps=5000 --val-steps=500 --epochs=10 --batch-size=1 --experiment-tag="NO_RET_REG_NO_RET_CLAS" \
    --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_pos_exp" --comet-workspace="$COMET_WORKSPACE" \
    --no-snapshots \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="transformer_classification,fusionnet_regression" \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/train.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/val.csv --no-evaluation \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt

    # no transformer output, no transformer validation, no retinanet classification, no retinanet regression, no retinanet validation
    python3 train.py --steps=5000 --val-steps=500 --epochs=10 --batch-size=1 --experiment-tag="NO_TRANS_NO_RET_CLAS_NO_RET_REG" \
    --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_ablation_pos_exp" --comet-workspace="$COMET_WORKSPACE" \
    --no-snapshots \
    fusionnet $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/FusionMultimodalCSV/virtual/Annotations/val.csv \
    --output-filter="fusionnet_regression" \
    vision retinanet $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/train.csv \
    $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/classmaps.csv \
    --val-annotations $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/val.csv --no-evaluation \
    language_translation transformer $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --val-golden-set $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val_golden.txt --no-evaluation
  done

else
  echo "Activation failed - skipping python package installations. Expected \"$virtual_env_name\" but got $(if [[ -z "$VIRTUAL_ENV" ]]; then echo "nothing"; else basename "$VIRTUAL_ENV"; fi) instead"
fi

cd $scripts_dir
