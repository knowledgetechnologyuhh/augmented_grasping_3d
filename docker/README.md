
# Docker Environments

On updating the latest drivers, it may become difficult to install previous CUDA 
versions which are necessary for the execution of some of the old code snippets. We have 
created an nvidia-docker image to contain all the scripts and libraries needed. This container 
only runs on a linux host with a CUDA supported GPU.

To use your display within the docker container, run the following command on your host system:

```
xhost +local:docker
```

###build

Navigate to this directory and run the following command:
(copy you ssh private key into this directory under the name id_rsa)
```
docker build --force-rm -t augmented_grasping_3d .
```

### run

Navigate to the root of this repository and run the following command:

```
docker run -it \
    --runtime=nvidia \
    -e DISPLAY=$DISPLAY \
    -e COMET_API_KEY=$COMET_API_KEY \
    -e COMET_WORKSPACE=$COMET_API_KEY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $DS_PATH:/home/user/augmented_grasping_3d/datasets/ \
    --entrypoint "/bin/bash" \
    augmented_grasping_3d 
```