NNODES=3 \
NODE_RANK=0 \
NPROC_PER_NODE=8 \
MASTER_ADDR=10.231.139.69 \
MASTER_PORT=29500 \
JOB_NAME=sensenovau1_5_8b_pt \
load_optimizer=model \
auto_resume=false \
resume_ds=false \
bash /datasets/codes_zsqiao/SenseNova-U1/training/shell/train_u1/U1.5_8B.sh

# 续训
# NNODES=3 \
# NODE_RANK=0 \
# NPROC_PER_NODE=8 \
# MASTER_ADDR=10.231.139.69 \
# MASTER_PORT=29500 \
# JOB_NAME=sensenovau1_5_8b_pt \
# load_optimizer=all \
# auto_resume=true \
# resume_ds=true \
# bash training/shell/train_u1/U1.5_8B.sh
