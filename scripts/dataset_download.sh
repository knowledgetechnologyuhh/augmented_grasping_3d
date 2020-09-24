#!/bin/bash


scripts_dir=`pwd`
echo "Downloading dataset from kaggle.com"

# install kaggle
pip3 install kaggle

# setup all the packages
cd $DS_PATH
mkdir GeneratedData -p
cd GeneratedData

# download and unzip the dataset
kaggle datasets download fabawi/augmented-extended-train-robots
unzip augmented-extended-train-robots.zip
# delete the file when completed
# rm augmented-extended-train-robots.zip

# replace the dataset paths
cd $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/
sed -i 's#{DS_PATH}#'$DS_PATH'#g' val.csv
sed -i 's#{DS_PATH}#'$DS_PATH'#g' train.csv
sed -i 's#{DS_PATH}#'$DS_PATH'#g' all.csv

cd $DS_PATH/GeneratedData/SimVisionMultimodalCSV/virtual/Annotations/
sed -i 's#{DS_PATH}#'$DS_PATH'#g' val.csv
sed -i 's#{DS_PATH}#'$DS_PATH'#g' all.csv

cd $DS_PATH/GeneratedData/SpeechMultimodalCSV/virtual/Annotations/
sed -i 's#{DS_PATH}#'$DS_PATH'#g' val.csv
sed -i 's#{DS_PATH}#'$DS_PATH'#g' train.csv
sed -i 's#{DS_PATH}#'$DS_PATH'#g' all.csv

# run the python script for postprocessing images (make sure you have atleast python3.5 installed)
cd $DS_PATH/GeneratedData/VisionMultimodalCSV/virtual/Annotations/
python3 annotation_postprocessor
python3 annotation_postprocessor --crop

cd $scripts_dir
