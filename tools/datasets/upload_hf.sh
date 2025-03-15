#!/bin/sh
#$ -cwd
#$ -l cpu_160=1
#$ -l h_rt=0:24:00:00
#$ -o outputs/$JOB_ID
#$ -e outputs/$JOB_ID
#$ -p -5

# priority: -5: normal, -4: high, -3: highest

cd /gs/fs/tga-bayes-crest/taishi/workspace/maxtext
source venv/bin/activate
export HUGGINGFACE_TOKEN="$1"
python tools/datasets/upload_hf.py --files "$2" --repo "$3" --format "$4"

