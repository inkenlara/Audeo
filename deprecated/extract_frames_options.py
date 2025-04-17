# Option 1: for VP9/H.264/AVC
def option_1(video_path, output_dir, coords):
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


# option 2: for av1
def extract_frames(video_path, output_dir, coords):
    """Extract and crop frames using ffmpeg (works with AV1 videos)"""
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    # Get video resolution using ffprobe
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        width, height = map(int, result.stdout.strip().split(','))
    except Exception as e:
        print(f"Failed to probe video {video_path}: {e}")
        return 0

    print(f"Detected video resolution: {width}x{height}")
    print(f"Original crop coords: {coords}")

    x1, y1, x2, y2 = coords
    # Build ffmpeg command
    output_pattern = os.path.join(output_dir, "frame_%06d.jpg")
    ffmpeg_cmd = [
        "ffmpeg", "-v", "error", "-c:v", "libdav1d",
        "-i", video_path,
        "-vf", f"crop={crop_w}:{crop_h}:{x1}:{y1}",
        "-q:v", "2",  # good quality JPEG
        output_pattern
    ]

    print("Running ffmpeg to extract frames...")
    try:
        subprocess.run(ffmpeg_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed: {e}")
        return 0

    frame_count = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
    print(f"Extracted {frame_count} frames from {video_path}")
    return frame_count