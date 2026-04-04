import os
import pandas as pd
import json
import numpy as np
import torch
import pickle
from transformers import AutoTokenizer, AutoModel
import re

def prepare_data():
    project_dir = "/home/user01/aiotlab/pndhuy/CLS/GloFE"
    data_dir = "/home/user01/aiotlab/sondinh/SLR_Neurips/CSLR_dataset/Group1"
    output_dir = os.path.join(project_dir, "notebooks/cslr-v1.0")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Download and save Tokenizer
    tokenizer_path = os.path.join(output_dir, "cslr-bpe-tokenizer")
    print("Saving PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    tokenizer.save_pretrained(tokenizer_path)
    
    # 2. Process Vocabulary
    print("Processing vocabulary...")
    vocab_df = pd.read_csv(os.path.join(data_dir, "Vocabulary_lab.csv"))
    unique_vns = set()
    sentence_to_vns = {}
    
    for idx, row in vocab_df.iterrows():
        sentence_id = row['ID_sentence']
        sign_sentence = str(row['Sign_sentence'])
        
        # Extract signs separated by | or /
        signs = [s.strip() for s in re.split(r'[|/]', sign_sentence)]
        # Clean up punctuation if needed, removing trailing ? . etc.
        cleaned_signs = []
        for s in signs:
            # We preserve exactly what was extracted, maybe strip punctuation
            clean_s = s.strip().strip('?.!,').strip()
            if clean_s:
                cleaned_signs.append(clean_s)
                unique_vns.add(clean_s)
                
        sentence_to_vns[sentence_id] = cleaned_signs

    unique_vns = sorted(list(unique_vns))
    
    # Write VN indices
    vn_idxs_path = os.path.join(output_dir, "cslr_VN_idxs.txt")
    with open(vn_idxs_path, 'w', encoding='utf-8') as f:
        for idx, vn in enumerate(unique_vns):
            f.write(f"{idx} {vn}\n")
    
    # 3. Process Dataset mapping
    print("Processing dataset mapping...")
    ds_df = pd.read_csv(os.path.join(data_dir, "Dataset_group1_lab.csv"))
    
    vid_to_vns = {}
    tsv_data = []
    
    # Group by video_belong (which we can use as unique video sequence, but dataset has Front/Left/Right)
    # The user mentioned each mp4 will be extracted by MMPose. 
    # Video name format: 0001_group1_00001_front_lab_001.mp4 
    for idx, row in ds_df.iterrows():
        video_path = row['Sentence_video_path']
        vid = os.path.basename(video_path).replace('.mp4', '')
        sentence_id = row['ID_sentence']
        raw_text = row['Sentence']
        
        # VN matched mapping
        vid_to_vns[vid] = sentence_to_vns.get(sentence_id, [])
        
        tsv_data.append({
            'vid': vid,
            'raw-text': raw_text
        })
        
    with open(os.path.join(output_dir, "cslr_VN_matched.json"), 'w', encoding='utf-8') as f:
        json.dump(vid_to_vns, f, ensure_ascii=False, indent=2)
        
    # Split 80/10/10
    print("Creating splits...")
    df_tsv = pd.DataFrame(tsv_data)
    # Use random state for reproducible splits
    df_tsv = df_tsv.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df_tsv)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    df_tsv.loc[:train_end-1, 'split'] = 'train'
    df_tsv.loc[train_end:val_end-1, 'split'] = 'valid'
    df_tsv.loc[val_end:, 'split'] = 'test'
    
    # reorder columns to match openasl
    df_tsv = df_tsv[['split', 'vid', 'raw-text']]
    df_tsv.to_csv(os.path.join(data_dir, "cslr-v1.0.tsv"), sep='\t', index=False)
    
    # 4. Generate Embeddings using PhoBERT
    print("Generating embeddings...")
    model = AutoModel.from_pretrained("vinai/phobert-base")
    model.eval()
    
    # PhoBERT dimension is 768
    embed_dim = 768
    vocab_size = len(unique_vns)
    embeddings = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    
    with torch.no_grad():
        for i, vn in enumerate(unique_vns):
            # PhoBERT expects text without word segmentation unless explicitly specified, 
            # we will tokenize using our tokenizer
            inputs = tokenizer(vn, return_tensors="pt")
            outputs = model(**inputs)
            # Pool the embeddings (mean of last hidden state)
            cls_emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings[i] = cls_emb
    
    with open(os.path.join(output_dir, "cslr_VN_embed.pkl"), 'wb') as f:
        pickle.dump(embeddings, f)
        
    print("Data preparation complete!")

if __name__ == "__main__":
    prepare_data()
