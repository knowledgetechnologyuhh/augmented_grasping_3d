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


### In this experiment, we try different embeddings on the transformer network
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
	--config $scripts_dir/transformer_embeddings_config/config_emb_10.ini --steps=10 --batch-size=1 --epochs=1 \
	--experiment-tag="SANITY_CHECK_TEST" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"


  # perform 3 rounds
  for i in {1..3}
  do
    echo "ROUND $i \n\n"
    # default initialization (size 10)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --config $scripts_dir/transformer_embeddings_config/config_emb_10.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="DEFAULT_10" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  
    # default initialization (size 50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --config $scripts_dir/transformer_embeddings_config/config_emb_50.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="DEFAULT_50" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  
    # default initialization (size 100)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --config $scripts_dir/transformer_embeddings_config/config_emb_100.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="DEFAULT_100" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  
    # default initialization (size 200)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --config $scripts_dir/transformer_embeddings_config/config_emb_200.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="DEFAULT_200" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  
    # default initialization (size 500)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --config $scripts_dir/transformer_embeddings_config/config_emb_500.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="DEFAULT_500" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  
    # default initialization (size 1000)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --config $scripts_dir/transformer_embeddings_config/config_emb_1000.ini --batch-size=60 --steps=500 --epochs=32 \
    --experiment-tag="DEFAULT_1000" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  
  
    # word2vec cbow initialization (size 10)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_cbow_source_10.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_cbow_target_10.npy \
    --config $scripts_dir/transformer_embeddings_config/config_emb_10.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="W2V_CBOW_10" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  
    # word2vec cbow initialization (size 50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_cbow_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_cbow_target_50.npy \
    --config $scripts_dir/transformer_embeddings_config/config_emb_50.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="W2V_CBOW_50" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  
    # word2vec cbow initialization (size 100)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_cbow_source_100.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_cbow_target_100.npy \
    --config $scripts_dir/transformer_embeddings_config/config_emb_100.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="W2V_CBOW_100" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  
    # word2vec cbow initialization (size 200)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_cbow_source_200.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_cbow_target_200.npy \
    --config $scripts_dir/transformer_embeddings_config/config_emb_200.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="W2V_CBOW_200" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  
    # word2vec cbow initialization (size 500)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_cbow_source_500.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_cbow_target_500.npy \
    --config $scripts_dir/transformer_embeddings_config/config_emb_500.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="W2V_CBOW_500" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  
    # word2vec cbow initialization (size 1000)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_cbow_source_1000.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_cbow_target_1000.npy \
    --config $scripts_dir/transformer_embeddings_config/config_emb_1000.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="W2V_CBOW_1000" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  
  
    # word2vec skipgram initialization (size 10)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_10.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_10.npy \
    --config $scripts_dir/transformer_embeddings_config/config_emb_10.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="W2V_SG_10" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  
    # word2vec skipgram initialization (size 50)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_50.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_50.npy \
    --config $scripts_dir/transformer_embeddings_config/config_emb_50.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="W2V_SG_50" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  
    # word2vec skipgram initialization (size 100)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_100.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_100.npy \
    --config $scripts_dir/transformer_embeddings_config/config_emb_100.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="W2V_SG_100" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  
    # word2vec skipgram initialization (size 200)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_200.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_200.npy \
    --config $scripts_dir/transformer_embeddings_config/config_emb_200.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="W2V_SG_200" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  
    # word2vec skipgram initialization (size 500)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_500.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_500.npy \
    --config $scripts_dir/transformer_embeddings_config/config_emb_500.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="W2V_SG_500" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  
    # word2vec skipgram initialization (size 1000)
    python3 train.py $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/train.csv \
    --val-annotations $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/val.csv \
    --vocab $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/vocab.lst \
    --i-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_source_1000.npy \
    --o-embedding-matrix $DS_PATH/GeneratedData/LanguageTranslationMultimodalCSV/virtual/Annotations/w2v_sg_target_1000.npy \
    --config $scripts_dir/transformer_embeddings_config/config_emb_1000.ini --batch-size=60 --steps=27000 --epochs=32 \
    --experiment-tag="W2V_SG_1000" --comet-api-key="$COMET_API_KEY" --comet-project-name="ir_transformer_emb_exp" --comet-workspace="$COMET_WORKSPACE"
  done

else
  echo "Activation failed - skipping experiments. Expected \"$virtual_env_name\" but got $(if [[ -z "$VIRTUAL_ENV" ]]; then echo "nothing"; else basename "$VIRTUAL_ENV"; fi) instead"
fi

cd $scripts_dir
