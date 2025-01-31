# TPU Swallow Gemma2

## 環境構築

jaxをすべてのVMにインストールをします

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=all \
    --command="pip install \
    --upgrade 'jax[tpu]>0.3.0' \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
```

## checkpoint convert

flax形式のcheckpointをダウンロードをしてきます

```bash
curl -L -u $KAGGLE_USERNAME:$KAGGLE_KEY \
  -o /mnt/filestore/checkpoints/model.tar.gz \
  https://www.kaggle.com/api/v1/models/google/gemma-2/flax/gemma2-2b/1/download
```

```bash
tar -xf  model.tar.gz
```

maxtext形式へconvertします
それぞれのVMインスタンスで実行をすると、書き込みが重複するのと、1VM内で実行できるように、CPUを指定します
他に良い方法がある場合は教えてください

```bash
export JAX_PLATFORM_NAME=cpu
```

読み込みと書き込みのパスは適宜変更をしてください

```bash
python MaxText/convert_gemma2_chkpt.py \
    --base_model_path /mnt/filestore/checkpoints/gemma2-2b \
    --maxtext_model_path /mnt/filestore/checkpoints_maxtext/gemma2-2b \
    --model_size 2b
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

```bash
gcloud compute tpus tpu-vm create ${TPU_NAME} \
  --zone=${ZONE} \
  --accelerator-type=v4-64 \
  --version=tpu-ubuntu2204-base
```

作り直したあと、TPUが見えるかどうかを確認してください


## 参考ドキュメント

https://cloud.google.com/filestore/docs/service-tiers?hl=ja
https://cloud.google.com/tpu/docs/v4?hl=ja
