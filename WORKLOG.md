# TPU Swallow Gemma2

```bash
export TPU_NAME=
export ZONE=
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
    --command="sudo mkdir -p /mnt/filestore && sudo mount -t nfs 10.77.37.202:/swallow /mnt/filestore && echo '10.77.37.202:/swallow /mnt/filestore nfs defaults 0 0' | sudo tee -a /etc/fstab"
```

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="sudo chmod -R 777 /mnt/filestore"
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

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="cd maxtext && bash setup.sh"
```

wandbもloginする


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

## 参考

以下のようなエラーが出るときは、

```
RuntimeError: Unable to initialize backend 'tpu': INTERNAL: Failed to get global TPU topology. (set JAX_PLATFORMS='' to automatically choose an available backend)
```

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


## 参考ドキュメント

https://cloud.google.com/filestore/docs/service-tiers?hl=ja
https://cloud.google.com/tpu/docs/v4?hl=ja
https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/tree/main/tpu_info