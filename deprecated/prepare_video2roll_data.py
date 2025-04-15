import os
import shutil
from pathlib import Path
import json
import pickle

def load_video_mapping():
    """Load the video ID to title mapping"""
    with open('video_mapping.json', 'r') as f:
        return json.load(f)

def get_labeled_frames(label_file):
    """Get the set of frame numbers that have labels"""
    with open(label_file, 'rb') as f:
        labels = pickle.load(f)
    return set(labels.keys())

def copy_labeled_frames(input_dir, output_dir, labeled_frames):
    """Copy only frames that exist in the label dictionary"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all frame files and sort them by frame number
    frame_files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith('.jpg')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    
    for frame_file in frame_files:
        frame_num = int(frame_file.split('_')[1].split('.')[0])
        if frame_num in labeled_frames:
            shutil.copy2(
                os.path.join(input_dir, frame_file),
                os.path.join(output_dir, frame_file)
            )

def main():
    # Load video mapping
    video_mapping = load_video_mapping()
    
    # Create directory structure
    base_dir = 'data'
    os.makedirs(os.path.join(base_dir, 'input_images', 'training'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'input_images', 'testing'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'labels', 'training'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'labels', 'testing'), exist_ok=True)
    
    # Process training data
    train_videos = [d for d in os.listdir('data/frames/training') 
                   if os.path.isdir(os.path.join('data/frames/training', d))]
    
    for video_id in train_videos:
        if video_id not in video_mapping:
            print(f"Warning: No mapping found for video {video_id}")
            continue
            
        video_info = video_mapping[video_id]
        if video_info['section'] != 'train':
            continue
            
        label_file_name = video_info['label_file']
        output_dir = os.path.join(base_dir, 'input_images', 'training', label_file_name[:-4])  # Remove .pkl
        
        # Skip if output directory already exists
        if os.path.exists(output_dir):
            print(f"Skipping {video_id} - output directory already exists")
            continue
            
        print(f"Processing training video: {video_id} -> {label_file_name}")
        
        # Get frames that have labels
        label_file = os.path.join('data_provided/labels/training', label_file_name)
        if not os.path.exists(label_file):
            print(f"Warning: Label file not found for {label_file_name}")
            continue
            
        labeled_frames = get_labeled_frames(label_file)
        print(f"Found {len(labeled_frames)} labeled frames")
        
        # Copy only frames that have labels
        input_dir = os.path.join('data/frames/training', video_id)
        copy_labeled_frames(input_dir, output_dir, labeled_frames)
        
        # Copy label file
        output_label = os.path.join(base_dir, 'labels', 'training', label_file_name)
        shutil.copy2(label_file, output_label)
    
    # Process testing data
    test_videos = [d for d in os.listdir('data/frames/testing') 
                  if os.path.isdir(os.path.join('data/frames/testing', d))]
    
    for video_id in test_videos:
        if video_id not in video_mapping:
            print(f"Warning: No mapping found for video {video_id}")
            continue
            
        video_info = video_mapping[video_id]
        if video_info['section'] != 'test':
            continue
            
        label_file_name = video_info['label_file']
        output_dir = os.path.join(base_dir, 'input_images', 'testing', label_file_name[:-4])  # Remove .pkl
        
        # Skip if output directory already exists
        if os.path.exists(output_dir):
            print(f"Skipping {video_id} - output directory already exists")
            continue
            
        print(f"Processing testing video: {video_id} -> {label_file_name}")
        
        # Get frames that have labels
        label_file = os.path.join('data_provided/labels/testing', label_file_name)
        if not os.path.exists(label_file):
            print(f"Warning: Label file not found for {label_file_name}")
            continue
            
        labeled_frames = get_labeled_frames(label_file)
        print(f"Found {len(labeled_frames)} labeled frames")
        
        # Copy only frames that have labels
        input_dir = os.path.join('data/frames/testing', video_id)
        copy_labeled_frames(input_dir, output_dir, labeled_frames)
        
        # Copy label file
        output_label = os.path.join(base_dir, 'labels', 'testing', label_file_name)
        shutil.copy2(label_file, output_label)

if __name__ == "__main__":
    main() 