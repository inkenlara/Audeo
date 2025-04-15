import os
import numpy as np
import cv2
from tqdm import tqdm
import yt_dlp
import pickle
from pathlib import Path
import shutil
from piano_coords import train_piano_coords, test_piano_coords
import time
import subprocess
import json

# Configuration
VIDEO_FPS = 25
AUDIO_SAMPLE_RATE = 16000
FRAME_DURATION = 2  # seconds
FRAMES_PER_SEGMENT = FRAME_DURATION * VIDEO_FPS

# Create necessary directories
os.makedirs('data/videos', exist_ok=True)
os.makedirs('data/frames', exist_ok=True)
os.makedirs('data/midi', exist_ok=True)
os.makedirs('data/labels', exist_ok=True)

def create_video_mapping():
    """Create mapping between video IDs and their titles"""
    mapping = {}
    
    # Read video IDs from Video_Id.md
    with open('Video_Id.md', 'r') as f:
        lines = f.readlines()
    
    current_section = None
    for line in lines:
        line = line.strip()
        if line.startswith('# Training'):
            current_section = 'train'
        elif line.startswith('# Testing'):
            current_section = 'test'
        elif line.startswith('- https://youtu.be/'):
            # Extract video ID from the URL
            video_id = line.split('/')[-1].strip()
            if not video_id:  # Skip empty lines
                continue
            mapping[video_id] = {
                'section': current_section,
                'label_file': f"{line.split('#')[1].strip()}.pkl" if '#' in line else f"{video_id}.pkl"
            }
    
    # Save mapping to file
    with open('video_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)
    
    return mapping

def extract_frames(video_path, output_dir, coords):
    """Extract and crop frames from video"""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return 0
        
    # Try to open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return 0
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Original crop coordinates: {coords}")
    
    # Get video ID from path
    video_id = os.path.basename(video_path).split('.')[0]
    
    # Special handling for videos with different aspect ratios
    if video_id in ['cAnmwgC-JRw', 'DMdJLEGrUrg', 'w77mBaWOOh0', 'xXwCryMItHs']:
        # These videos need different scaling factors
        scale_x = width / 1920.0
        scale_y = height / 1080.0
                
        x1, y1, x2, y2 = coords
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
    else:
        # Normal scaling for other videos
        scale_x = width / 1920.0
        scale_y = height / 1080.0
        
        x1, y1, x2, y2 = coords
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
    
    print(f"Scaled crop coordinates: ({x1}, {y1}, {x2}, {y2})")
    
    # Ensure coordinates are within bounds
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    
    if x1 >= x2 or y1 >= y2:
        print(f"Invalid crop region: ({x1}, {y1}, {x2}, {y2})")
        return 0
    
    frame_count = 0
    success = True
    
    while success:
        success, frame = cap.read()
        if not success:
            break
            
        # Check if frame is valid
        if frame is None or frame.size == 0:
            print(f"Invalid frame {frame_count} from {video_path}")
            continue
            
        try:
            # Crop frame using coordinates
            cropped_frame = frame[y1:y2, x1:x2]
            
            # Check if cropped frame is valid
            if cropped_frame is None or cropped_frame.size == 0:
                print(f"Invalid cropped frame {frame_count} from {video_path}")
                continue
            
            # Save frame
            frame_path = os.path.join(output_dir, f'frame_{frame_count:06d}.jpg')
            cv2.imwrite(frame_path, cropped_frame)
            frame_count += 1
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Extracted {frame_count} frames from {video_path}")
                
        except Exception as e:
            print(f"Error processing frame {frame_count} from {video_path}: {str(e)}")
            continue
        
    cap.release()
    print(f"Successfully extracted {frame_count} frames from {video_path}")
    return frame_count

def download_videos(video_ids, output_dir):
    """Download videos from YouTube using yt-dlp"""
    ydl_opts = {
        'format': 'bestvideo[height<=1080][ext=mp4][vcodec=avc1]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4][vcodec=avc1]/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'retries': 10,
        'fragment_retries': 10,
        'extract_flat': False,
        'prefer_ffmpeg': True,
        'merge_output_format': 'mp4',
    }
    
    for video_id in tqdm(video_ids, desc="Downloading videos"):
        output_path = os.path.join(output_dir, f"{video_id}.mp4")
        
        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"Video {video_id} already exists, skipping...")
            continue
            
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print(f"Downloading video {video_id}...")
                ydl.download([f'https://youtube.com/watch?v={video_id}'])
                print(f"Successfully downloaded video {video_id}")
        except Exception as e:
            print(f"Error downloading {video_id}: {str(e)}")
            # Wait a bit before trying the next video
            time.sleep(5)

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

def prepare_final_dataset(video_mapping):
    """Prepare the final dataset with only labeled frames"""
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

def main():
    # Create video mapping
    print("Creating video mapping...")
    video_mapping = create_video_mapping()
    
    # Get video IDs from Video_Id.md
    with open('Video_Id.md', 'r') as f:
        lines = f.readlines()
    
    train_videos = []
    test_videos = []
    current_section = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('# Training'):
            current_section = 'train'
        elif line.startswith('# Testing'):
            current_section = 'test'
        elif line.startswith('- https://youtu.be/'):
            # Extract video ID from the URL
            video_id = line.split('/')[-1].strip()
            if not video_id:  # Skip empty lines
                continue
            if current_section == 'train':
                train_videos.append(video_id)
            elif current_section == 'test':
                test_videos.append(video_id)
    
    print(f"Found {len(train_videos)} training videos and {len(test_videos)} testing videos")
    
    # Download and process videos
    if train_videos:
        print("Downloading training videos...")
        download_videos(train_videos, 'data/videos/train')
    else:
        print("No training videos found!")
    
    if test_videos:
        print("Downloading testing videos...")
        download_videos(test_videos, 'data/videos/test')
    else:
        print("No testing videos found!")
    
    # Process videos
    for video_id, coords in zip(train_videos, train_piano_coords):
        video_path = f'data/videos/train/{video_id}.mp4'
        frames_dir = f'data/frames/train/{video_id}'
        os.makedirs(frames_dir, exist_ok=True)
        
        print(f"Processing training video {video_id}...")
        frame_count = extract_frames(video_path, frames_dir, coords)
        print(f"Extracted {frame_count} frames from video {video_id}")
    
    for video_id, coords in zip(test_videos, test_piano_coords):
        video_path = f'data/videos/test/{video_id}.mp4'
        frames_dir = f'data/frames/test/{video_id}'
        os.makedirs(frames_dir, exist_ok=True)
        
        print(f"Processing testing video {video_id}...")
        frame_count = extract_frames(video_path, frames_dir, coords)
        print(f"Extracted {frame_count} frames from video {video_id}")
    
    # Prepare final dataset
    print("\nPreparing final dataset...")
    prepare_final_dataset(video_mapping)
    
    print("\nDataset preparation complete!")

if __name__ == '__main__':
    main() 