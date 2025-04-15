import os
import cv2
import numpy as np
import subprocess
from Midi_synth import MIDISynth
import re

def create_video(frames_dir, output_path, fps=25):
    """
    Create a video from frames using OpenCV.
    
    Args:
        frames_dir: Directory containing the frames (should be named sequentially)
        output_path: Where to save the video
        fps: Frames per second (default 25)
    """
    # Get all frame files and sort them
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
    
    if not frame_files:
        raise ValueError(f"No JPG files found in {frames_dir}")
    
    # Get the first frame to determine video dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    print(f"Creating video from {len(frame_files)} frames...")
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frames_dir, frame_file))
        out.write(frame)
    
    out.release()
    print(f"Successfully created video: {output_path}")
    return output_path

def add_audio_to_video(video_path, audio_file, output_path):
    """
    Add audio to an existing video using ffmpeg.
    
    Args:
        video_path: Path to the video file
        audio_file: Path to the audio file
        output_path: Where to save the final video with audio
    """
    print("Adding audio to video...")
    cmd = [
        'ffmpeg',
        '-i', video_path,  # Input video
        '-i', audio_file,  # Input audio
        '-c:v', 'copy',    # Copy video stream
        '-c:a', 'aac',     # Encode audio as AAC
        '-shortest',       # Finish encoding when the shortest input stream ends
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully created video with audio: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error combining video and audio: {e}")

def get_safe_filename(original_name, set_type, index):
    """
    Convert original video name to a safe filename.
    
    Args:
        original_name: Original video name
        set_type: 'training' or 'testing'
        index: Index number for the video
    """
    return f"{set_type}_no{index}"

def natural_sort_key(s):
    """
    Key function for natural sorting of strings containing numbers.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def process_video(frames_dir, set_type, index, output_dir):
    """
    Process a single video: synthesize audio and create video with audio.
    
    Args:
        frames_dir: Directory containing the frames
        set_type: 'training' or 'testing'
        index: Index number for the video
        output_dir: Directory to save the output
    """
    # Get the video name from the frames directory
    video_name = os.path.basename(frames_dir)
    safe_name = get_safe_filename(video_name, set_type, index)
    
    print(f"\nProcessing {video_name}...")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
    
    # Step 1: Synthesize audio
    print("Synthesizing audio...")
    midi_folder = f"data_provided/midi/{set_type}/"
    midi_synth = MIDISynth(midi_folder, video_name, "Acoustic Grand Piano", midi=True)
    midi_synth.GetNote()
    midi_synth.Synthesize()
    
    # Step 2: Create video without audio
    video_only_path = os.path.join(output_dir, "videos", f"{safe_name}_video_only.mp4")
    create_video(frames_dir, video_only_path)
    
    # Step 3: Add audio to video
    audio_file = os.path.join("data", "synthesized", "midi", video_name, f"Midi-{video_name}-Acoustic Grand Piano.wav")
    video_with_audio_path = os.path.join(output_dir, "videos", f"{safe_name}.mp4")
    add_audio_to_video(video_only_path, audio_file, video_with_audio_path)
    
    # Clean up temporary video-only file
    if os.path.exists(video_only_path):
        os.remove(video_only_path)
        print(f"Removed temporary video-only file: {video_only_path}")

def main():
    # Base directories
    input_base_dir = "data/input_images"
    output_base_dir = "data/generated"
    
    # Process training videos
    training_dir = os.path.join(input_base_dir, "training")
    if os.path.exists(training_dir):
        training_videos = sorted([d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))], 
                               key=natural_sort_key)
        for i, video_dir in enumerate(training_videos, 1):
            frames_dir = os.path.join(training_dir, video_dir)
            process_video(frames_dir, "training", i, output_base_dir)
    
    # Process testing videos
    testing_dir = os.path.join(input_base_dir, "testing")
    if os.path.exists(testing_dir):
        testing_videos = sorted([d for d in os.listdir(testing_dir) if os.path.isdir(os.path.join(testing_dir, d))], 
                              key=natural_sort_key)
        for i, video_dir in enumerate(testing_videos, 1):
            frames_dir = os.path.join(testing_dir, video_dir)
            process_video(frames_dir, "testing", i, output_base_dir)

if __name__ == "__main__":
    main() 