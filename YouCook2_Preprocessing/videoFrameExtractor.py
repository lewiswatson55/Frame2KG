import os
import argparse
import subprocess

def extract_frames(video_path, output_folder, fps=1):
    """
    Extracts frames from a video using ffmpeg at the specified fps.
    Frames are saved as 'frame_0001.jpg', 'frame_0002.jpg', etc.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Construct the ffmpeg command
    output_pattern = os.path.join(output_folder, "frame_%04d.jpg")
    command = [
        "ffmpeg", "-i", video_path,  # Input video
        "-vf", f"fps={fps}",         # Extract frames at specified FPS
        output_pattern               # Output file pattern
    ]

    # Run ffmpeg
    subprocess.run(command, check=True)
    print(f"Frames saved in '{output_folder}'")


def process_videos(input_dir, output_dir, fps=1):
    """
    Recursively walks through 'input_dir' to find all video files.
    Reproduces the folder structure under 'output_dir' and
    saves frames for each video in a matching subfolder named
    after the video file (minus its extension).
    """
    # Extensions you consider as videos
    valid_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

    # Walk through the entire directory structure
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Check if file has a valid video extension
            _, ext = os.path.splitext(file)
            if ext.lower() in valid_extensions and not file.startswith("._"):
                # Full path to the input video
                full_video_path = os.path.join(root, file)

                # Calculate the relative path (so we can replicate structure)
                rel_path = os.path.relpath(root, input_dir)

                # The folder to hold frames will be:
                # output_dir/<rel_path>/<video_name_without_ext>/
                video_name = os.path.splitext(file)[0]
                output_folder = os.path.join(output_dir, rel_path, video_name)

                # Extract frames
                print(f"Processing: {full_video_path}")
                extract_frames(full_video_path, output_folder, fps=fps)


def main():
    parser = argparse.ArgumentParser(
        description="Recursively extract frames from videos in a directory, "
                    "replicating the directory structure at the output location."
    )
    parser.add_argument("-i", "--input_dir", required=True,
                        help="Path to the input directory containing subfolders/videos.")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Path to the output directory where frames will be stored.")
    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second to extract (default=1).")

    args = parser.parse_args()

    # Process videos in the specified directories
    process_videos(args.input_dir, args.output_dir, args.fps)


if __name__ == "__main__":
    main()
