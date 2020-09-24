#!/bin/bash

# arg1: (Default:venv3) name of the virtual environment (will be stored in the main directory of this project). Default is stored in the file ./venv . if problems occur, change the name in the file

# get the virtual environment name from the first argument to the script. if none is given, use the last virtual environment.
virtual_env_name=${1:-$(<venv)}
rm ./venv
echo "$virtual_env_name">>./venv
echo "Requested virtual environment: $virtual_env_name"


# get dir
dir=`dirname "$BASH_SOURCE"`
scripts_dir=`pwd`
virtual_env_cmd="virtualenv"
project_dir="$(dirname "$scripts_dir")"
echo "The project directory: " $project_dir

# create the virtual environment
cd $project_dir
echo "Checking for virtualenv:"
if [[ -d "$virtual_env_name" ]]; then
    echo "Existing virtualenv found"
else
  echo "No virtualenv found - setting up new virtualenv"
  # Test for virtualenv
  if ! [[ -x "$(command -v virtualenv)" ]]; then
    #pip install --user virtualenv
    pip3 install virtualenv
    virtual_env_cmd="./usr/local/bin/virtualenv"
  fi
  $virtual_env_cmd -p /usr/bin/python3 $virtual_env_name --system-site-packages # /usr/bin/python3.5 works but breaks the dataset_generation requirements (specifically random.choices() in noisemix). python3.6 is incompatible with the docker env
fi
echo "Activating virtualenv"
source $virtual_env_name"/bin/activate"

# inside the virtual environment, run all setups
if [[ $(basename "$VIRTUAL_ENV") == $virtual_env_name ]]; then
  echo "Checking python packages"
  # docker tf1.12 py3.5 environment requirements
  pip3 install 'scikit-learn==0.21.3' 'scikit-image<0.15' 'keras==2.2.4'
  # tf 1.10 disables setuptools upgrade which results in google-cloud-texttospeech not installing correctly. 1.12 has topological sorting problems. choose whatever is best for the task
  #pip3 install tensorflow-gpu==1.12
  #pip3 install keras==2.2.4
  cd $project_dir/modules/language_translation/transformer/
  python3 setup.py install
  cd $project_dir/modules/vision_object_detection/retinanet/
  python3 setup.py build_ext --inplace
  python3 setup.py install
  cd $project_dir/modules/data_fusion/fusionnet/
  python3 setup.py install
  #cd $project_dir/block_world/
  #python3 setup.py install
else
  echo "Activation failed - skipping python package installations. Expected \"$virtual_env_name\" but got $(if [[ -z "$VIRTUAL_ENV" ]]; then echo "nothing"; else basename "$VIRTUAL_ENV"; fi) instead"
fi

cd $scripts_dir
