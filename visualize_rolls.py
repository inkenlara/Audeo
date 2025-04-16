import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

def create_roll_visualization(ax, roll, title, start_frame, end_frame):
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
    
    # Add grid lines for better readability
    ax.grid(True, alpha=0.3, color='gray', linestyle='--')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def create_overlay_visualization(ax, roll1, roll2, title, start_frame, end_frame):
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

def visualize_all(gt_roll, est_roll, piece_name, file_stem, save_path=None):
    """Create a single figure with three subplots stacked vertically"""
    print(f"Creating combined visualization: {piece_name} - {file_stem}")
    
    # Get frame range from filename
    start_frame, end_frame = get_frame_range(file_stem)
    
    # Create figure with main title first
    fig = plt.figure(figsize=(15, 15), facecolor='white')
    fig.suptitle(f"{piece_name} - {file_stem}", y=0.98, fontsize=14)
    
    # Create subplots
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)
    
    # Create ground truth visualization
    create_roll_visualization(ax1, gt_roll, "Ground Truth", start_frame, end_frame)
    
    # Create estimated visualization
    create_roll_visualization(ax2, est_roll, "Estimated", start_frame, end_frame)
    
    # Create overlay visualization
    create_overlay_visualization(ax3, gt_roll, est_roll, "Overlay", start_frame, end_frame)
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        print(f"Saving to: {save_path}")
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def process_directory(gt_dir, est_dir, output_dir):
    """Process all matching files in the directories"""
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
        print(f"Created output directory: {piece_output_dir}")
        
        # Process each file in the piece directory
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
                print("Loading ground truth data...")
                gt_roll = load_npz_file(gt_file, is_ground_truth=True)
                print("Loading estimated data...")
                est_roll = load_npz_file(est_file, is_ground_truth=False)
                
                # Create combined visualization
                output_path = piece_output_dir / f"{gt_file.stem}_combined.png"
                print(f"Creating visualization for {gt_file.stem}...")
                visualize_all(
                    gt_roll,
                    est_roll,
                    piece_name,
                    gt_file.stem,
                    output_path
                )
                print(f"Successfully created visualization at {output_path}")
            except Exception as e:
                print(f"Error processing {gt_file}: {str(e)}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    gt_dir = "data_provided/midi/testing"
    est_dir = "data/estimate_Roll/testing"
    output_dir = "data/generated/rolls"
    
    print("Starting visualization process...")
    process_directory(gt_dir, est_dir, output_dir)
    print("Visualization process completed.") 