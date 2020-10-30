FROM tensorflow/tensorflow:1.12.0-rc2-devel-gpu-py3

RUN apt-get update && apt-get install -y python3.5-dev python-pydot python-pydot-ng graphviz llvm
RUN apt-get  install -y python-tk libsm6 libxext6 libxrender-dev

# install git
RUN apt-get install -y git apt-utils
RUN  apt-get -yqq install ssh
RUN mkdir /root/.ssh/
ADD id_rsa /root/.ssh/id_rsa
#RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa
RUN chmod 700 /root/.ssh/id_rsa
RUN chown -R root:root /root/.ssh
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN mkdir /home/user
WORKDIR /home/user

# RUN git clone --recurse-submodules git@github.com:knowledgetechnologyUHH/augmented_grasping_3d.git augmented_grasping_3d
COPY ./ ../augmented_grasping_3d/
ENV LANG C.UTF-8
RUN cd augmented_grasping_3d/scripts/ && ./setup_experiment_requirements.sh
