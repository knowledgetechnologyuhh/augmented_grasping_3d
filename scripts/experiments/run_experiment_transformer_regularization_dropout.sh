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


### In this experiment, we try different dropout probabilities on the transformer layers
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
  cd $project_dir/modules/language_translation/transformer/keras_transformer/bin/
  # this is a test script
  python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
  --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
  --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
  --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_cbow_source_10.npy \
  --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
  --config $scripts_dir/transformer_embeddings_config/config_emb_10.ini --steps=10 --batch-size=1 --epochs=1 \
  --experiment-tag="SANITY_CHECK_TEST" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_reg_exp" --comet-workspace="$COMET_WORKSPACE"

  # perform 3 rounds
  for i in {1..3}
  do
    echo "ROUND $i \n\n"
    # dropout 0.0 with the full dataset
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_regularization_config/config_drop_0.ini --batch-size=30 --steps=27000 --epochs=32 \
    --experiment-tag="FULL_DROP_0.0" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_reg_exp" --comet-workspace="$COMET_WORKSPACE"

    # dropout 0.1 with the full dataset
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_regularization_config/config_drop_0.1.ini --batch-size=30 --steps=27000 --epochs=32 \
    --experiment-tag="FULL_DROP_0.1" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_reg_exp" --comet-workspace="$COMET_WORKSPACE"

    # dropout 0.15 with the full dataset
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_regularization_config/config_drop_0.15.ini --batch-size=30 --steps=27000 --epochs=32 \
    --experiment-tag="FULL_DROP_0.15" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_reg_exp" --comet-workspace="$COMET_WORKSPACE"

    # dropout 0.2 with the full dataset
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_regularization_config/config_drop_0.2.ini --batch-size=30 --steps=27000 --epochs=32 \
    --experiment-tag="FULL_DROP_0.2" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_reg_exp" --comet-workspace="$COMET_WORKSPACE"

    # dropout 0.25 with the full dataset
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_regularization_config/config_drop_0.25.ini --batch-size=30 --steps=27000 --epochs=32 \
    --experiment-tag="FULL_DROP_0.25" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_reg_exp" --comet-workspace="$COMET_WORKSPACE"

    # dropout 0.3 with the full dataset
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_regularization_config/config_drop_0.3.ini --batch-size=30 --steps=27000 --epochs=32 \
    --experiment-tag="FULL_DROP_0.3" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_reg_exp" --comet-workspace="$COMET_WORKSPACE"

    # dropout 0.4 with the full dataset
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_regularization_config/config_drop_0.4.ini --batch-size=30 --steps=27000 --epochs=32 \
    --experiment-tag="FULL_DROP_0.4" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_reg_exp" --comet-workspace="$COMET_WORKSPACE"

    # dropout 0.5 with the full dataset
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_regularization_config/config_drop_0.5.ini --batch-size=30 --steps=27000 --epochs=32 \
    --experiment-tag="FULL_DROP_0.5" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_reg_exp" --comet-workspace="$COMET_WORKSPACE"
  done

else
  echo "Activation failed - skipping experiments. Expected \"$virtual_env_name\" but got $(if [[ -z "$VIRTUAL_ENV" ]]; then echo "nothing"; else basename "$VIRTUAL_ENV"; fi) instead"
fi

cd $scripts_dir
