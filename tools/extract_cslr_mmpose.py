# Specialized MMPose extraction for CSLR Dataset
import os
import warnings
from argparse import ArgumentParser
import cv2
import mmcv
import numpy as np
import pickle as pkl
from tqdm import tqdm

from mmpose.apis import (inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def main():
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('--pose-config', default='configs/hrnet_w48_coco_wholebody_384x288_dark.py', help='Config file for pose')
    parser.add_argument('--pose-checkpoint', default='checkpoints/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth', help='Checkpoint file for pose')
    parser.add_argument('--data-root', type=str, default='/home/user01/aiotlab/sondinh/SLR_Neurips/CSLR_dataset/Group1', help='Root of CSLR Group 1')
    parser.add_argument('--output-root', type=str, default='/home/user01/aiotlab/sondinh/SLR_Neurips/CSLR_dataset/Group1/mmpose', help='Output root for .pkl files')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--det-cat-id', type=int, default=1, help='Category id for person detection')
    parser.add_argument('--bbox-thr', type=float, default=0.3, help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument('--tsv-path', type=str, default=None, help='TSV file with video paths')
    parser.add_argument('--no-det', action='store_true', help='Skip person detection and use full frame')
    parser.add_argument('--sid', type=int, default=0, help='Split ID')
    parser.add_argument('--splits', type=int, default=1, help='Total splits')

    args = parser.parse_args()
    if not args.no_det:
        assert has_mmdet, 'Please install mmdet.'

    # Find all mp4 files and build map
    print(f'Scanning {args.data_root} for videos...')
    vid_map = {}
    for root, dirs, files in os.walk(args.data_root):
        for file in files:
            if file.endswith('.mp4'):
                vid_map[os.path.splitext(file)[0]] = os.path.join(root, file)

    all_videos = []
    if args.tsv_path and os.path.exists(args.tsv_path):
        import pandas as pd
        df = pd.read_csv(args.tsv_path, sep='\t')
        vid_col = 'vid' if 'vid' in df.columns else df.columns[0]
        for v in df[vid_col].tolist():
            v_name = os.path.splitext(os.path.basename(str(v)))[0]
            if v_name in vid_map:
                all_videos.append(vid_map[v_name])
            else:
                # Handle cases where full path is already in TSV
                v_path = os.path.join(args.data_root, str(v))
                if os.path.exists(v_path):
                    all_videos.append(v_path)
    else:
        all_videos = list(vid_map.values())
    
    all_videos = sorted(list(set(all_videos)))
    total_samples = len(all_videos)
    print(f'Total videos discovered: {total_samples}')

    os.makedirs(args.output_root, exist_ok=True)

    # Split for parallel processing
    chunk = (total_samples + args.splits - 1) // args.splits
    current_videos = all_videos[args.sid * chunk : min((args.sid + 1) * chunk, total_samples)]
    print(f'Processing split {args.sid}/{args.splits}: {len(current_videos)} videos')

    # Initialize models
    print('Initializing models...')
    det_model = None
    if not args.no_det:
        det_model = init_detector(args.det_config, args.det_checkpoint, device=args.device.lower())
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info:
        dataset_info = DatasetInfo(dataset_info)
    dataset_type = pose_model.cfg.data['test']['type']

    arg_dict = {
        'det_model': det_model,
        'pose_model': pose_model,
        'dataset': dataset_type,
        'dataset_info': dataset_info,
        'output_root': args.output_root,
        'args': args,
    }

    for video_path in tqdm(current_videos):
        video_name = os.path.basename(video_path).replace('.mp4', '')
        output_path = os.path.join(args.output_root, f'{video_name}.pkl')
        
        if os.path.exists(output_path):
            continue
            
        process_video(video_path, output_path, arg_dict)

def process_video(video_path, output_path, arg_dict):
    det_model = arg_dict['det_model']
    pose_model = arg_dict['pose_model']
    dataset = arg_dict['dataset']
    dataset_info = arg_dict['dataset_info']
    args = arg_dict['args']

    video = mmcv.VideoReader(video_path)
    if not video.opened:
        print(f'Failed to open {video_path}')
        return

    results = []
    for frame_id, cur_frame in enumerate(video):
        if not args.no_det:
            mmdet_results = inference_detector(det_model, cur_frame)
            person_results = process_mmdet_results(mmdet_results, args.det_cat_id)
        else:
            # Use the entire frame as a bounding box [x1, y1, x2, y2, score]
            h, w = cur_frame.shape[:2]
            person_results = [{'bbox': np.array([0, 0, w, h, 1.0])}]

        # inference_top_down_pose_model returns list of pose_results
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            cur_frame,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info)

        if len(pose_results) > 0:
            # Take the first (highest confidence) person
            results.append(pose_results[0]['keypoints'])
        else:
            # Append zero keypoints if no person found to maintain timing
            # Assuming 133 keypoints for wholebody
            results.append(np.zeros((133, 3)))

    results = np.array(results)
    with open(output_path, 'wb') as f:
        pkl.dump(results, f)

if __name__ == '__main__':
    main()
