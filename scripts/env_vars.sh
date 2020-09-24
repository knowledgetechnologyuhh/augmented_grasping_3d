# the dataset path is necessary when running experiments (this path contains the GeneratedData directory)
export DS_PATH={YOUR_DS_PATH_HERE}
# these variables are machine specific so check if you need to change them in the first place
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export COMET_API_KEY=""
export COMET_WORKSPACE=""