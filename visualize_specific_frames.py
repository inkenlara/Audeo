import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

def load_npz_file(npz_path, is_ground_truth=True):
    """Load data from npz file"""
    print(f"Loading npz file: {npz_path}")
    data = np.load(str(npz_path))
    if is_ground_truth:
        # For ground truth, use the 'midi' array
        if 'midi' in data:
            roll = data['midi']
        else:
            raise ValueError(f"No 'midi' array found in {npz_path}")
    else:
        # For estimated, use the first array
        roll = data[data.files[0]]
    
    print(f"Roll shape: {roll.shape}")
    return roll

def get_frame_range(filename):
    """Extract frame range from filename (e.g., '648-698.npz' -> (648, 698))"""
    base = Path(filename).stem
    start, end = map(int, base.split('-'))
    return start, end

def create_roll_visualization(ax, roll, title, start_frame, end_frame, highlight_frame=None):
    """Create a single roll visualization in the given axis"""
    # Transpose the roll to have time on x-axis and pitch on y-axis
    roll = roll.T
    
    # Create a binary mask for the notes
    binary_roll = (roll > 0).astype(float)
    
    # Create RGB image with custom colors
    rgb_roll = np.ones((*binary_roll.shape, 3))  # Start with white background
    if "Ground Truth" in title:
        # Blue for ground truth
        rgb_roll[binary_roll > 0, 2] = 0.8  # Blue channel
        rgb_roll[binary_roll > 0, 0:2] = 0.2  # Darken other channels
    else:
        # Pink for estimated
        rgb_roll[binary_roll > 0, 0] = 0.8  # Red channel
        rgb_roll[binary_roll > 0, 2] = 0.8  # Blue channel
        rgb_roll[binary_roll > 0, 1] = 0.2  # Green channel
    
    ax.imshow(rgb_roll, aspect='auto', origin='lower')
    ax.set_title(title, pad=10)
    
    # Set x-axis to show actual frame numbers
    frame_numbers = np.arange(start_frame, end_frame + 1)
    # Calculate tick positions and labels
    tick_positions = np.arange(0, roll.shape[1], 10)
    tick_labels = frame_numbers[::10]
    
    # Ensure we have the same number of positions and labels
    min_length = min(len(tick_positions), len(tick_labels))
    tick_positions = tick_positions[:min_length]
    tick_labels = tick_labels[:min_length]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Pitch')
    
    # Add vertical line for highlighted frame if specified
    if highlight_frame is not None:
        frame_idx = highlight_frame - start_frame
        if 0 <= frame_idx < roll.shape[1]:
            ax.axvline(x=frame_idx, color='red', linestyle='--', linewidth=2)
    
    # Add grid lines for better readability
    ax.grid(True, alpha=0.3, color='gray', linestyle='--')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def create_overlay_visualization(ax, roll1, roll2, title, start_frame, end_frame, highlight_frame=None):
    """Create an overlay visualization in the given axis"""
    # Transpose both rolls
    roll1 = roll1.T
    roll2 = roll2.T
    
    # Ensure both rolls have the same shape
    min_shape = (min(roll1.shape[0], roll2.shape[0]), min(roll1.shape[1], roll2.shape[1]))
    roll1 = roll1[:min_shape[0], :min_shape[1]]
    roll2 = roll2[:min_shape[0], :min_shape[1]]
    
    # Create binary masks
    binary_roll1 = (roll1 > 0).astype(float)
    binary_roll2 = (roll2 > 0).astype(float)
    
    # Create RGB image with white background
    overlay = np.ones((min_shape[0], min_shape[1], 3))
    
    # Ground truth only (blue)
    gt_only = binary_roll1 * (1 - binary_roll2)
    overlay[gt_only > 0, 2] = 0.8  # Blue channel
    overlay[gt_only > 0, 0:2] = 0.2  # Darken other channels
    
    # Estimated only (pink)
    est_only = binary_roll2 * (1 - binary_roll1)
    overlay[est_only > 0, 0] = 0.8  # Red channel
    overlay[est_only > 0, 2] = 0.8  # Blue channel
    overlay[est_only > 0, 1] = 0.2  # Green channel
    
    # Overlap (green)
    overlap = binary_roll1 * binary_roll2
    overlay[overlap > 0, 1] = 0.8  # Green channel
    overlay[overlap > 0, 0] = 0.2  # Red channel
    overlay[overlap > 0, 2] = 0.2  # Blue channel
    
    ax.imshow(overlay, aspect='auto', origin='lower')
    ax.set_title(title, pad=10)
    
    # Set x-axis to show actual frame numbers
    frame_numbers = np.arange(start_frame, end_frame + 1)
    # Calculate tick positions and labels
    tick_positions = np.arange(0, min_shape[1], 10)
    tick_labels = frame_numbers[::10]
    
    # Ensure we have the same number of positions and labels
    min_length = min(len(tick_positions), len(tick_labels))
    tick_positions = tick_positions[:min_length]
    tick_labels = tick_labels[:min_length]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Pitch')
    
    # Add vertical line for highlighted frame if specified
    if highlight_frame is not None:
        frame_idx = highlight_frame - start_frame
        if 0 <= frame_idx < min_shape[1]:
            ax.axvline(x=frame_idx, color='red', linestyle='--', linewidth=2)
    
    # Add grid lines
    ax.grid(True, alpha=0.3, color='gray', linestyle='--')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Create custom legend with matching colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0.2, 0.2, 0.8), label='Ground Truth'),
        Patch(facecolor=(0.8, 0.2, 0.8), label='Estimated'),
        Patch(facecolor=(0.2, 0.8, 0.2), label='Overlap')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

def create_frame_visualization(ax, gt_frame, est_frame, title):
    """Create a visualization of a specific frame's data"""
    print(f"Creating frame visualization - GT shape: {gt_frame.shape}, Est shape: {est_frame.shape}")
    print(f"GT values: min={np.min(gt_frame)}, max={np.max(gt_frame)}, mean={np.mean(gt_frame)}")
    print(f"Est values: min={np.min(est_frame)}, max={np.max(est_frame)}, mean={np.mean(est_frame)}")
    
    # Create a figure with two bars side by side
    width = 0.35
    x = np.arange(len(gt_frame))
    
    # Plot the raw values
    ax.bar(x - width/2, gt_frame, width, color=(0.2, 0.2, 0.8), label='Ground Truth')
    ax.bar(x + width/2, est_frame, width, color=(0.8, 0.2, 0.8), label='Estimated')
    
    ax.set_title(title, pad=10)
    ax.set_xlabel('Pitch')
    ax.set_ylabel('Value')
    
    # Set x-axis ticks to show pitch numbers
    pitch_ticks = np.arange(0, len(gt_frame), 12)  # Show every octave
    ax.set_xticks(pitch_ticks)
    ax.set_xticklabels([f'Pitch {p}' for p in pitch_ticks])
    
    ax.grid(True, alpha=0.3, color='gray', linestyle='--')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add legend
    ax.legend(loc='upper right')

def load_image_frame(frame_number, piece_name):
    """Load the corresponding image frame"""
    # Construct the path to the image frame
    # Images are in data/input_images/testing/[piece_name]/frame_[number].jpg
    # Frame numbers are padded with zeros (e.g., frame_006471.jpg)
    padded_frame = f"{frame_number:06d}"
    image_path = Path(f"data/input_images/testing/{piece_name}/frame_{padded_frame}.jpg")
    if not image_path.exists():
        print(f"Warning: Image not found at {image_path}")
        return None
    return plt.imread(str(image_path))

def visualize_frame(gt_roll, est_roll, piece_name, file_stem, frame_number, save_path=None):
    """Create a single figure with four subplots stacked vertically for a specific frame"""
    print(f"Creating visualization for frame {frame_number}: {piece_name} - {file_stem}")
    
    # Get frame range from filename
    start_frame, end_frame = get_frame_range(file_stem)
    
    # Create figure with main title first
    fig = plt.figure(figsize=(15, 20), facecolor='white')
    fig.suptitle(f"{piece_name} - {file_stem} - Frame {frame_number}", y=0.98, fontsize=14)
    
    # Create subplots
    ax1 = plt.subplot(4, 1, 1)
    ax2 = plt.subplot(4, 1, 2)
    ax3 = plt.subplot(4, 1, 3)
    ax4 = plt.subplot(4, 1, 4)
    
    # Create ground truth visualization
    create_roll_visualization(ax1, gt_roll, "Ground Truth", start_frame, end_frame, frame_number)
    
    # Create estimated visualization
    create_roll_visualization(ax2, est_roll, "Estimated", start_frame, end_frame, frame_number)
    
    # Create overlay visualization
    create_overlay_visualization(ax3, gt_roll, est_roll, "Overlay", start_frame, end_frame, frame_number)
    
    # Load and display the image frame
    image = load_image_frame(frame_number, piece_name)
    if image is not None:
        ax4.imshow(image)
        ax4.set_title(f"Frame {frame_number} Image")
        ax4.axis('off')
    else:
        ax4.text(0.5, 0.5, f"Image not found: frame_{frame_number}.jpg", ha='center', va='center')
        ax4.set_title(f"Frame {frame_number} Image")
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        print(f"Saving to: {save_path}")
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def process_directory(gt_dir, est_dir, output_dir, num_frames_per_piece=5):
    """Process all matching files in the directories and create visualizations for random frames"""
    gt_dir = Path(gt_dir)
    est_dir = Path(est_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Processing ground truth directory: {gt_dir}")
    print(f"Processing estimated directory: {est_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process each piece
    for piece_dir in gt_dir.iterdir():
        if not piece_dir.is_dir():
            continue
            
        piece_name = piece_dir.name
        print(f"\nProcessing {piece_name}")
        
        # Create output directory for this piece
        piece_output_dir = output_dir / piece_name
        piece_output_dir.mkdir(exist_ok=True)
        
        # Collect all frames from all files in this piece
        all_frames = []
        for gt_file in piece_dir.glob('*.npz'):
            print(f"\nFound ground truth file: {gt_file}")
            # Find corresponding estimated file
            est_file = est_dir / piece_name / gt_file.name
            if not est_file.exists():
                print(f"Warning: No matching estimated file found for {gt_file}")
                continue
                
            print(f"Found matching estimated file: {est_file}")
            
            try:
                # Load data
                gt_roll = load_npz_file(gt_file, is_ground_truth=True)
                est_roll = load_npz_file(est_file, is_ground_truth=False)
                
                # Get frame range
                start_frame, end_frame = get_frame_range(gt_file.stem)
                
                # Add frames from this file to the collection
                all_frames.extend([(frame, gt_roll, est_roll, gt_file.stem) 
                                 for frame in range(start_frame, end_frame + 1)])
                
            except Exception as e:
                print(f"Error processing {gt_file}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Select random frames from the entire piece
        if all_frames:
            selected_frames = random.sample(all_frames, min(num_frames_per_piece, len(all_frames)))
            
            # Create visualization for each selected frame
            for frame_number, gt_roll, est_roll, file_stem in selected_frames:
                output_path = piece_output_dir / f"{file_stem}_frame_{frame_number}.png"
                visualize_frame(
                    gt_roll,
                    est_roll,
                    piece_name,
                    file_stem,
                    frame_number,
                    output_path
                )

if __name__ == "__main__":
    gt_dir = "data_provided/midi/testing"
    est_dir = "data/estimate_Roll/testing"
    output_dir = "data/generated/specific_frames"
    
    print("Starting visualization process...")
    process_directory(gt_dir, est_dir, output_dir)
    print("Visualization process completed.") 