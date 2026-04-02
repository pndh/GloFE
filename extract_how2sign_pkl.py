import os
import json
import numpy as np
import pickle
from tqdm import tqdm
from multiprocessing import Pool

INPUT_DIR = "data/How2Sign/features/openpose_output/json"
OUTPUT_DIR = "data/How2Sign/features/cache_inst"

def process_video(video_id):
    video_dir = os.path.join(INPUT_DIR, video_id)
    if not os.path.isdir(video_dir):
        return None
    
    json_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.json')])
    if not json_files:
        return None
    
    # Store all frames
    all_frames = []
    
    for j_file in json_files:
        jf_path = os.path.join(video_dir, j_file)
        with open(jf_path, 'r') as f:
            data = json.load(f)
            
        if not data.get('people'):
             # Empty frame padding
             all_frames.append(np.zeros((137, 3), dtype=np.float32))
             continue
             
        person = data['people'][0] # Take the main person
        
        # 1. Pose (25)
        pose_2d = person.get('pose_keypoints_2d', [])
        if len(pose_2d) > 0:
            pose = np.array(pose_2d).reshape(-1, 3)[:25]
        else:
            pose = np.zeros((25, 3))
            
        # 2. Hand Left (21)
        hl_2d = person.get('hand_left_keypoints_2d', [])
        if len(hl_2d) > 0:
            hand_left = np.array(hl_2d).reshape(-1, 3)[:21]
        else:
            hand_left = np.zeros((21, 3))
            
        # 3. Hand Right (21)
        hr_2d = person.get('hand_right_keypoints_2d', [])
        if len(hr_2d) > 0:
            hand_right = np.array(hr_2d).reshape(-1, 3)[:21]
        else:
            hand_right = np.zeros((21, 3))
            
        # 4. Face (70)
        face_2d = person.get('face_keypoints_2d', [])
        if len(face_2d) > 0:
            face = np.array(face_2d).reshape(-1, 3)[:70]
        else:
            face = np.zeros((70, 3))
            
        # Concatenate 25 + 21 + 21 + 70 = 137
        frame_kp = np.concatenate([pose, hand_left, hand_right, face], axis=0).astype(np.float32)
        all_frames.append(frame_kp)
        
    frames_array = np.stack(all_frames, axis=0) # [T, 137, 3]
    
    out_file = os.path.join(OUTPUT_DIR, f"{video_id}.pkl")
    with open(out_file, 'wb') as f:
        pickle.dump(frames_array, f)
        
    return video_id

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_ids = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    print(f"Processing {len(video_ids)} videos...")
    
    with Pool(processes=8) as pool:
        for _ in tqdm(pool.imap_unordered(process_video, video_ids), total=len(video_ids)):
            pass
    print("Done extracting features to PKL!")
