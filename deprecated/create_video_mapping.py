import os
import json
from pathlib import Path
import re

def get_piece_number(filename):
    """Extract the number from the Bach piece filename"""
    match = re.search(r'No\.(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def main():
    # Read video IDs from Video_Id.md
    with open('Video_Id.md', 'r') as f:
        lines = f.readlines()
    
    video_ids = []
    current_section = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('# Training'):
            current_section = 'train'
        elif line.startswith('# Testing'):
            current_section = 'test'
        elif line.startswith('- https://youtu.be/'):
            video_id = line.split('/')[-1].strip()
            if video_id:
                video_ids.append((video_id, current_section))
    
    # Get list of label files and sort them numerically
    train_labels = sorted(
        [f for f in os.listdir('data_provided/labels/training') if f.endswith('.pkl')],
        key=get_piece_number
    )
    test_labels = sorted(
        [f for f in os.listdir('data_provided/labels/testing') if f.endswith('.pkl')],
        key=get_piece_number
    )
    
    # Create mapping
    video_info = {}
    
    # Map training videos
    train_videos = [v for v in video_ids if v[1] == 'train']
    for (video_id, section), label_file in zip(train_videos, train_labels):
        video_info[video_id] = {
            'id': video_id,
            'label_file': label_file,
            'section': section
        }
    
    # Map testing videos
    test_videos = [v for v in video_ids if v[1] == 'test']
    for (video_id, section), label_file in zip(test_videos, test_labels):
        video_info[video_id] = {
            'id': video_id,
            'label_file': label_file,
            'section': section
        }
    
    # Save mapping to JSON file
    with open('video_mapping.json', 'w') as f:
        json.dump(video_info, f, indent=2)
    
    print(f"Created mapping for {len(video_info)} videos")
    print("Mapping saved to video_mapping.json")
    
    # Print the mapping in a readable format
    print("\nVideo ID to Label File Mapping:")
    for video_id, info in video_info.items():
        print(f"{video_id} -> {info['label_file']}")

if __name__ == '__main__':
    main() 