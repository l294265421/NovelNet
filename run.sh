#!/usr/bin/env bash
start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

python=/data/miniconda3/bin/python

export LANG="zh_CN.UTF-8"

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

# tar -zcf session-rec-pytorch-private.tar run.sh algorithms evaluation helper preprocessing run_config.py common
${python} run_config.py $@

end_time=`date +%Y%m%d%H%M%S`
echo ${end_time}
