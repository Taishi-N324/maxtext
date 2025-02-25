# TPU Swallow Gemma2

```bash
export TPU_NAME=
export ZONE=
export WANDB_PASSWORD=
export FILESTORE_IP=
```

特定のVMインスタンスにsshする場合

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME}  --zone=${ZONE} --worker=${i}
```

## 環境構築

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="sudo apt-get update && sudo apt-get install -y nfs-common"
```

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="sudo apt-get update && sudo apt-get install -y nfs-kernel-server"
```

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="sudo modprobe nfs && sudo modprobe nfsd"
```

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="sudo mkdir -p /mnt/filestore && sudo mount -t nfs ${FILESTORE_IP}:/swallow /mnt/filestore && echo '${FILESTORE_IP}:/swallow /mnt/filestore nfs defaults 0 0' | sudo tee -a /etc/fstab"
```

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="pip install \
    --upgrade 'jax[tpu]>0.3.0' \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
```

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="git clone https://github.com/Taishi-N324/maxtext.git && cd maxtext && git switch swallow"
```


TPU v6eでは先に以下のコマンドを実行する必要があります。

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="pip install 'setuptools==67.8.0'"
```

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="cd maxtext && bash setup.sh"
```

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \\
    --zone=${ZONE} \\
    --worker=all \\
    --command="echo 'machine api.wandb.ai\
  login user\
  password ${WANDB_PASSWORD}' > ~/.netrc"
```


gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="echo 'machine api.wandb.ai
  login user
  password ${WANDB_PASSWORD}' > ~/.netrc"

`scripts/environment/v6e_install.sh`　を実行することでここまでの環境構築を実行できます

## checkpoint convert

flax形式のcheckpointをダウンロードをしてきます

```bash
export KAGGLE_USERNAME=
export KAGGLE_KEY=
```

```bash
curl -L -u $KAGGLE_USERNAME:$KAGGLE_KEY \
  -o /mnt/filestore/checkpoints/model.tar.gz \
  https://www.kaggle.com/api/v1/models/google/gemma-2/flax/gemma2-2b/1/download
```

9Bもダウンロードできます

```bash
curl -L -u $KAGGLE_USERNAME:$KAGGLE_KEY \
  -o /mnt/filestore/checkpoints/model.tar.gz \
  https://www.kaggle.com/api/v1/models/google/gemma-2/flax/gemma2-9b/1/download
```

27B

```bash
curl -L -u $KAGGLE_USERNAME:$KAGGLE_KEY \
  -o /mnt/filestore/checkpoints/model.tar.gz \
  https://www.kaggle.com/api/v1/models/google/gemma-2/flax/gemma2-27b/1/download
```

```bash
tar -xf  model.tar.gz
```

maxtext形式へconvertします

cpuに指定をし、すべてのtpuでの実行待ち or すべてのworkerからの書き込みを防ぎます

```bash
export JAX_PLATFORMS=cpu
export JAX_THREADED_P_MAP=false
```

読み込みと書き込みのパスは適宜変更をしてください

```bash
python MaxText/convert_gemma2_chkpt.py \
  --base_model_path /mnt/filestore/checkpoints/gemma2-2b \
  --maxtext_model_path /mnt/filestore/checkpoints_maxtext/gemma2-2b \
  --model_size 2b
```

```bash
python MaxText/convert_gemma2_chkpt.py \
  --base_model_path /mnt/filestore/checkpoints/gemma2_9b_pt \
  --maxtext_model_path /mnt/filestore/checkpoints_maxtext/gemma2-9b \
  --model_size 9b
```

```bash
python MaxText/convert_gemma2_chkpt.py \
  --base_model_path /mnt/filestore/checkpoints/27bpt/gemma2_27b_pt \
  --maxtext_model_path /mnt/filestore/checkpoints_maxtext/gemma2-27b \
  --model_size 27b
```

### HF形式への checkpoint convert

27Bのconvertに必要なCPUメモリが ~524GBなので、v6e-4での作業をお勧めします

torchのインストールをします

```bash
pip install torch
```

下記のスクリプトを参考にしてください

```
scripts/ckpt_convert/gemma2_9b/exp2.sh
scripts/ckpt_convert/gemma2_2b/exp2.sh
```

## 参考

以下のようなエラーが出るときは、

```
RuntimeError: Unable to initialize backend 'tpu': INTERNAL: Failed to get global TPU topology. (set JAX_PLATFORMS='' to automatically choose an available backend)
```

## TPU v4

`/dev/accel*`で

```
/dev/accel0
/dev/accel1
/dev/accel2
/dev/accel3
```
が見えるかどうかを確認します

存在しない場合はVMを作り直します

参考 https://github.com/jax-ml/jax/issues/13260

```bash
gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE}
```

accelerator-typeは適宜変更しましょう

```bash
gcloud compute tpus tpu-vm create ${TPU_NAME} \
  --zone=${ZONE} \
  --accelerator-type=v4-64 \
  --version=tpu-ubuntu2204-base
```

作り直したあと、TPUが見えるかどうかを確認してください

TPUのプロセスが残ると、新規のプロセスが走らなくなります

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="sudo lsof -w /dev/accel0 | awk '{print \$2}' | tail -n +2 | xargs -r sudo kill -9"
```

参考 https://www.googlecloudcommunity.com/gc/Developer-Tools/TPU-POD-no-initiation/m-p/597179

## TPU v6

v6でVMを作成する時は、`v2-alpha-tpuv6e`を使用します : ref https://cloud.google.com/tpu/docs/runtimes?hl=ja#tensorflow

spotでの利用でない場合はspotは不要です

```bash
gcloud compute tpus tpu-vm create ${TPU_NAME} \
  --zone=${ZONE} \
  --accelerator-type=v6e-256 \
  --version=v2-alpha-tpuv6e \
  --spot
```

v4とは異なり、`/dev/vfio/{0,1,2,3}` にChipが存在します

```bash
pip install tpu-info
```

`tpu-info` コマンドでChipが見れます。

v6で tpuでのプロセスをkillするコマンドは以下です。

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="sudo lsof -w /dev/vfio/* | awk '{print \$2}' | tail -n +2 | xargs -r sudo kill -9"
```

## 参考ドキュメント

https://cloud.google.com/filestore/docs/service-tiers?hl=ja
https://cloud.google.com/tpu/docs/v4?hl=ja
https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/tree/main/tpu_info