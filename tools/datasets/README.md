# maxtext用のデータセットの準備

## text keyのみを取り出す

```bash
mkdir -p /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/
mkdir -p /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-wiki-top10/
mkdir -p /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-gemma-top10/
mkdir -p /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-gemma-top10-qa-filtered/
mkdir -p /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/dclm-baseline-1.0/sampling/
mkdir -p /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/bigcode/stack-v2_fujii_san/
mkdir -p /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/finemath/finemath-4plus-jsonl/
```

```bash
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/ja_wiki_merged.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/ja_wiki_merged.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/Swallow_v2/edu/filter-v2-wiki-top10/dump_0.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-wiki-top10/dump_0.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/Swallow_v2/edu/filter-v2-wiki-top10/dump_1.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-wiki-top10/dump_1.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/Swallow_v2/edu/filter-v2-wiki-top10/dump_2.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-wiki-top10/dump_2.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/Swallow_v2/edu/filter-v2-wiki-top10/dump_3.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-wiki-top10/dump_3.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/Swallow_v2/edu/filter-v2-wiki-top10/dump_4.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-wiki-top10/dump_4.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/Swallow_v2/edu/filter-v2-gemma-top10/dump_0.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-gemma-top10/dump_0.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/Swallow_v2/edu/filter-v2-gemma-top10/dump_1.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-gemma-top10/dump_1.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/Swallow_v2/edu/filter-v2-gemma-top10/dump_2.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-gemma-top10/dump_2.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/Swallow_v2/edu/filter-v2-gemma-top10/dump_3.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-gemma-top10/dump_3.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/Swallow_v2/edu/filter-v2-gemma-top10/dump_4.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-gemma-top10/dump_4.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgh-24IDU/filter-v2-gemma-top10-qa-filtered/dump_0.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-gemma-top10-qa-filtered/dump_0.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgh-24IDU/filter-v2-gemma-top10-qa-filtered/dump_1.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-gemma-top10-qa-filtered/dump_1.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgh-24IDU/filter-v2-gemma-top10-qa-filtered/dump_2.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-gemma-top10-qa-filtered/dump_2.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgh-24IDU/filter-v2-gemma-top10-qa-filtered/dump_3.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-gemma-top10-qa-filtered/dump_3.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgh-24IDU/filter-v2-gemma-top10-qa-filtered/dump_4.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/Swallow_v2/edu/filter-v2-gemma-top10-qa-filtered/dump_4.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/default_plain_text_format.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/default_plain_text_format.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/en_wiki_merged_train.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/en_wiki_merged_train.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/cosmopedia_automathtext_train.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/cosmopedia_automathtext_train.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/cosmopedia_khanacademy_train.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/cosmopedia_khanacademy_train.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/cosmopedia_openstax_train.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/cosmopedia_openstax_train.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/cosmopedia_stanford_train.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/cosmopedia_stanford_train.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/cosmopedia_stories_train.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/cosmopedia_stories_train.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgi-24IBB/datasets/Swallow/raw/cosmopedia/sampling/cosmopedia_web_samples_v1_train.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/sampling_cosmopedia_web_samples_v1_train.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgi-24IBB/datasets/Swallow/raw/cosmopedia/sampling/cosmopedia_web_samples_v2_train.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/sampling_cosmopedia_web_samples_v2_train.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tga-bayes-crest/Swallow/raw/cosmopedia_wikihow_train.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/cosmopedia_wikihow_train.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgi-24IBB/datasets/Swallow/raw/dclm-baseline-1.0/sampling/global-shard_01_of_10.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/dclm-baseline-1.0/sampling/global-shard_01_of_10.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgi-24IBB/datasets/Swallow/raw/dclm-baseline-1.0/sampling/global-shard_02_of_10.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/dclm-baseline-1.0/sampling/global-shard_02_of_10.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgi-24IBB/datasets/Swallow/raw/dclm-baseline-1.0/sampling/global-shard_03_of_10.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/dclm-baseline-1.0/sampling/global-shard_03_of_10.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgi-24IBB/datasets/Swallow/raw/dclm-baseline-1.0/sampling/global-shard_04_of_10.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/dclm-baseline-1.0/sampling/global-shard_04_of_10.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgi-24IBB/datasets/Swallow/raw/dclm-baseline-1.0/sampling/global-shard_05_of_10.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/dclm-baseline-1.0/sampling/global-shard_05_of_10.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgi-24IBB/datasets/Swallow/raw/dclm-baseline-1.0/sampling/global-shard_06_of_10.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/dclm-baseline-1.0/sampling/global-shard_06_of_10.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgi-24IBB/datasets/Swallow/raw/dclm-baseline-1.0/sampling/global-shard_07_of_10.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/dclm-baseline-1.0/sampling/global-shard_07_of_10.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgi-24IBB/datasets/Swallow/raw/dclm-baseline-1.0/sampling/global-shard_08_of_10.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/dclm-baseline-1.0/sampling/global-shard_08_of_10.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgi-24IBB/datasets/Swallow/raw/dclm-baseline-1.0/sampling/global-shard_09_of_10.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/dclm-baseline-1.0/sampling/global-shard_09_of_10.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgi-24IBB/datasets/Swallow/raw/dclm-baseline-1.0/sampling/global-shard_10_of_10.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/dclm-baseline-1.0/sampling/global-shard_10_of_10.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgi-24IBB/datasets/Swallow/raw/bigcode/stack-v2_fujii_san/merged_4.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/bigcode/stack-v2_fujii_san/merged_4.jsonl
qsub -g ${group_name} tools/datasets/jsonl_to_text.sh /gs/bs/tgh-24IDU/datasets/raw/pretrain/finemath/finemath-4plus-jsonl/finemath-4plus-merged.jsonl /gs/bs/tga-bayes-crest/Swallow/gemma_tpu_raw/finemath/finemath-4plus-jsonl/finemath-4plus-merged.jsonl
```
