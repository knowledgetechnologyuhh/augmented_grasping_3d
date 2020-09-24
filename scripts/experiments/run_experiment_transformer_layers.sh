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


### In this experiment, we vary a major number of the transformer's hyperparams
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
	--o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_cbow_target_10.npy \
	--config $scripts_dir/transformer_layers_config/config_lay_3_12_128.ini --steps=10 --batch-size=1 --epochs=1 \
	--experiment-tag="SANITY_CHECK_TEST" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"


  # perform 3 rounds
  for i in {1..3}
  do
    echo "ROUND $i \n\n"
    # layers 4_12_128 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_4_12_128.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="4_12_128" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 6_12_128 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_6_12_128.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="6_12_128" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 8_12_128 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_8_12_128.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="8_12_128" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 10_12_128 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_10_12_128.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="10_12_128" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"


    # layers 4_32_128 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_4_32_128.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="4_32_128" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 6_32_128 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_6_32_128.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="6_32_128" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 8_32_128 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_8_32_128.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="8_32_128" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 10_32_128 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_10_32_128.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="10_32_128" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"


    # layers 4_128_128 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_4_128_128.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="4_128_128" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 6_128_128 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_6_128_128.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="6_128_128" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 8_128_128 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_8_128_128.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="8_128_128" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 10_128_128 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_10_128_128.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="10_128_128" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"


    # layers 4_256_128 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_4_256_128.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="4_256_128" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 6_256_128 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_6_256_128.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="6_256_128" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 8_256_128 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_8_256_128.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="8_256_128" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 10_256_128 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_10_256_128.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="10_256_128" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"


    # layers 4_12_256 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_4_12_256.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="4_12_256" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 6_12_256 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_6_12_256.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="6_12_256" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 8_12_256 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_8_12_256.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="8_12_256" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 10_12_256 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_10_12_256.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="10_12_256" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"


    # layers 4_32_256 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_4_32_256.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="4_32_256" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 6_32_256 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_6_32_256.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="6_32_256" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 8_32_256 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_8_32_256.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="8_32_256" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 10_32_256 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_10_32_256.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="10_32_256" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"


    # layers 4_128_256 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_4_128_256.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="4_128_256" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 6_128_256 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_6_128_256.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="6_128_256" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 8_128_256 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_8_128_256.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="8_128_256" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 10_128_256 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_10_128_256.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="10_128_256" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"


    # layers 4_256_256 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_4_256_256.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="4_256_256" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 6_256_256 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_6_256_256.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="6_256_256" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 8_256_256 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_8_256_256.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="8_256_256" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 10_256_256 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_10_256_256.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="10_256_256" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"


    # layers 4_12_1024 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_4_12_1024.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="4_12_1024" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 6_12_1024 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_6_12_1024.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="6_12_1024" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 8_12_1024 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_8_12_1024.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="8_12_1024" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 10_12_1024 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_10_12_1024.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="10_12_1024" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"


    # layers 4_32_1024 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_4_32_1024.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="4_32_1024" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 6_32_1024 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_6_32_1024.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="6_32_1024" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 8_32_1024 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_8_32_1024.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="8_32_1024" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 10_32_1024 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_10_32_1024.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="10_32_1024" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"


    # layers 4_128_1024 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_4_128_1024.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="4_128_1024" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 6_128_1024 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_6_128_1024.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="6_128_1024" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 8_128_1024 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_8_128_1024.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="8_128_1024" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 10_128_1024 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_10_128_1024.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="10_128_1024" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"


    # layers 4_256_1024 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_4_256_1024.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="4_256_1024" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 6_256_1024 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_6_256_1024.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="6_256_1024" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 8_256_1024 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_8_256_1024.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="8_256_1024" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 10_256_1024 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_10_256_1024.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="10_256_1024" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"


    # layers 4_12_2048 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_4_12_2048.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="4_12_2048" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 6_12_2048 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_6_12_2048.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="6_12_2048" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 8_12_2048 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_8_12_2048.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="8_12_2048" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 10_12_2048 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_10_12_2048.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="10_12_2048" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"


    # layers 4_32_2048 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_4_32_2048.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="4_32_2048" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 6_32_2048 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_6_32_2048.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="6_32_2048" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 8_32_2048 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_8_32_2048.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="8_32_2048" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 10_32_2048 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_10_32_2048.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="10_32_2048" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"


    # layers 4_128_2048 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_4_128_2048.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="4_128_2048" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 6_128_2048 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_6_128_2048.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="6_128_2048" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 8_128_2048 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_8_128_2048.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="8_128_2048" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 10_128_2048 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_10_128_2048.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="10_128_2048" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"


    # layers 4_256_2048 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_4_256_2048.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="4_256_2048" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 6_256_2048 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_6_256_2048.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="6_256_2048" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 8_256_2048 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_8_256_2048.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="8_256_2048" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"

    # layers 10_256_2048 using w2v-sg init(50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_layers_config/config_lay_10_256_2048.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="10_256_2048" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_lay_exp" --comet-workspace="$COMET_WORKSPACE"
  done

else
  echo "Activation failed - skipping experiments. Expected \"$virtual_env_name\" but got $(if [[ -z "$VIRTUAL_ENV" ]]; then echo "nothing"; else basename "$VIRTUAL_ENV"; fi) instead"
fi

cd $scripts_dir
