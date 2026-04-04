#!/bin/bash
# Template for CSLR training
# Replace paths with your actual local paths if needed

GPUS=1 # Adjust based on your available GPUs
DATASET_ROOT="/home/user01/aiotlab/sondinh/SLR_Neurips/CSLR_dataset/Group1"

WORK_DIR_PREFIX="./work_dir"
WORK_DIR="cslr/group1_run1"
FEAT_PATH="$DATASET_ROOT/mmpose"
LABEL_PATH="$DATASET_ROOT/cslr-v1.0.tsv"

ARTIFACT_DIR="./notebooks/cslr-v1.0"
TOKENIZER="$ARTIFACT_DIR/cslr-bpe-tokenizer"
VN_MATCHED="$ARTIFACT_DIR/cslr_VN_matched.json"
VN_IDXS="$ARTIFACT_DIR/cslr_VN_idxs.txt"
VN_EMBED="$ARTIFACT_DIR/cslr_VN_embed.pkl"

# Note: inter_cl_vocab should match the number of unique VN words in your dataset.
# The prepare_cslr_data.py script will output this number. 
# For Group 1, we will set it based on the generated idxs file.
VN_VOCAB_SIZE=$(wc -l < $VN_IDXS)

python train_openasl_pose_DDP_inter_VN.py \
    --ngpus $GPUS \
    --work_dir_prefix "$WORK_DIR_PREFIX" \
    --work_dir "$WORK_DIR" \
    --bs 16 --ls 0.1 --epochs 100 \
    --save_every 5 \
    --clip_length 512 \
    --vocab_size 64001 \
    --feat_path "$FEAT_PATH" \
    --label_path "$LABEL_PATH" \
    --vn_matched_path "$VN_MATCHED" \
    --vn_idxs_path "$VN_IDXS" \
    --eos_token "</s>" \
    --tokenizer "$TOKENIZER" \
    --pose_backbone "PartedPoseBackbone" \
    --pe_enc --mask_enc --lr 1e-4 --dropout_dec 0.1 --dropout_enc 0.1 \
    --inter_cl --inter_cl_margin 0.4 --inter_cl_alpha 1.0 \
    --inter_cl_vocab $VN_VOCAB_SIZE \
    --inter_cl_we_path "$VN_EMBED"
