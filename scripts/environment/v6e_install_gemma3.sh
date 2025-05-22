#!/bin/bash

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="sudo apt-get update && sudo apt-get install -y nfs-common"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="sudo apt-get update && sudo apt-get install -y nfs-kernel-server"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="sudo modprobe nfs && sudo modprobe nfsd"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="sudo mkdir -p /mnt/filestore && sudo mount -t nfs ${FILESTORE_IP}:/swallow /mnt/filestore && echo '${FILESTORE_IP}:/swallow /mnt/filestore nfs defaults 0 0' | sudo tee -a /etc/fstab"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="pip install \
    --upgrade 'jax[tpu]==0.5.3' \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="git clone https://github.com/Taishi-N324/maxtext.git && cd maxtext && git switch gemma3-swallow"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="pip install 'setuptools==67.8.0'"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="cd maxtext && bash setup.sh"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="echo 'machine api.wandb.ai\
  login user\
    password ${WANDB_PASSWORD}' > ~/.netrc"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="pip install \
    --upgrade 'jax[tpu]==0.5.3' \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"


gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="ls /mnt/filestore/gemma3"