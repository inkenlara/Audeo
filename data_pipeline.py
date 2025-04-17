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

# Create necessary directories
os.makedirs('data_hq/videos', exist_ok=True)
os.makedirs('data_hq/frames', exist_ok=True)
os.makedirs('data_hq/midi', exist_ok=True)
os.makedirs('data_hq/labels', exist_ok=True)
os.makedirs('data_hq/input_images', exist_ok=True)

# TODO: only works for some videos, does not work for videos ['_3qnL9ddHuw', 'HB8-w5CvMls', 'PIS76X17Mf8', 'vHi3_k4XOrA']
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
    print(f"Crop coordinates: {coords}")
    
    # Use coordinates directly
    x1, y1, x2, y2 = coords
    
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
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
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
    print(f"\nDebugging label file: {label_file}")
    with open(label_file, 'rb') as f:
        labels = pickle.load(f)
    
    # Print some debug information
    print(f"Total number of labeled frames: {len(labels)}")
    if labels:
        first_key = next(iter(labels.keys()))
        print(f"First frame number: {first_key}")
        print(f"First frame data type: {type(first_key)}")
        print(f"First frame data structure: {labels[first_key]}")
    
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
    
    # Get video IDs from Video_Id.md
    with open('Video_Id.md', 'r') as f:
        lines = f.readlines()
    
    train_videos = []
    test_videos = []
    current_section = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('# Training'):
            current_section = 'training'
        elif line.startswith('# Testing'):
            current_section = 'testing'
        elif line.startswith('- https://youtu.be/'):
            # Extract video ID from the URL
            video_id = line.split('/')[-1].strip()
            if not video_id:  # Skip empty lines
                continue
            if current_section == 'training':
                train_videos.append(video_id)
            elif current_section == 'testing':
                test_videos.append(video_id)
    
    print(f"Found {len(train_videos)} training videos and {len(test_videos)} testing videos")
    
    # Download and process videos
    # if train_videos:
    #     print("Downloading training videos...")
    #     download_videos(train_videos, 'data_hq/videos/training')
    # else:
    #     print("No training videos found!")
    
    # if test_videos:
    #     print("Downloading testing videos...")
    #     download_videos(test_videos, 'data_hq/videos/testing')
    # else:
    #     print("No testing videos found!")
    
    # Extract frames from videos
    # for video_id, coords in zip(train_videos, train_piano_coords):
    #     video_path = f'data_hq/videos/training/{video_id}.mp4'
    #     frames_dir = f'data_hq/frames/training/{video_id}'
    #     os.makedirs(frames_dir, exist_ok=True)
        
    #     print(f"Processing training video {video_id}...")
    #     frame_count = extract_frames(video_path, frames_dir, coords)
    #     print(f"Extracted {frame_count} frames from video {video_id}")
    
    # for video_id, coords in zip(test_videos, test_piano_coords):
    #     video_path = f'data_hq/videos/testing/{video_id}.mp4'
    #     frames_dir = f'data_hq/frames/testing/{video_id}'
    #     os.makedirs(frames_dir, exist_ok=True)
        
    #     print(f"Processing testing video {video_id}...")
    #     frame_count = extract_frames(video_path, frames_dir, coords)
    #     print(f"Extracted {frame_count} frames from video {video_id}")


if __name__ == '__main__':
    main() 