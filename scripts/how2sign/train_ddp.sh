source /mnt/disk1/miniconda3/etc/profile.d/conda.sh
conda activate glofe
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
GPUS=1
torchrun --nnodes=1 --nproc_per_node=$GPUS train_how2_pose_DDP_inter_VN.py \
    --ngpus $GPUS \
    --work_dir_prefix "/mnt/disk4/pndhuy/CLS/GloFE/work_dir" \
    --work_dir "how2sign/train_test" \
    --bs 8 --ls 0.2 --epochs 400 \
    --save_every 5 \
    --clip_length 512 --vocab_size 23136 \
    --feat_path "/mnt/disk4/pndhuy/CLS/GloFE/data/How2Sign/features/cache_inst" \
    --label_path "/mnt/disk4/pndhuy/CLS/GloFE/data/How2Sign/how2sign_realigned_{split}.csv" \
    --eos_token "</s>" \
    --tokenizer "notebooks/how2sign/how2sign-bpe25000-tokenizer-uncased" \
    --pose_backbone "OPPartedPoseBackbone" \
    --pe_enc --mask_enc --lr 3e-4 --dropout_dec 0.1 --dropout_enc 0.1 \
    --inter_cl_margin 0.4 --inter_cl_alpha 1.0 \
    --inter_cl_vocab 2191 \
    --inter_cl_we_path "notebooks/how2sign/uncased_filtred_glove_VN_embed.pkl"