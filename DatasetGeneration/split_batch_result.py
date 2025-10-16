#!/usr/bin/env python3
"""
Split a batch result JSONL file into individual JSON files organized by video clip.

Usage:
    python split_batch_results.py batch_results.jsonl output_directory

This script takes the OpenAI batch output and organizes it into a folder structure:
    output_directory/
        video_clip_name_1/
            frame_0001.json
            frame_0002.json
            ...
        video_clip_name_2/
            frame_0001.json
            ...
"""
import os
import sys
import json
import argparse
from pathlib import Path


def split_batch_results(batch_results_file, output_dir):
    """
    Split batch results into individual files organized by video clip.

    Args:
        batch_results_file: Path to the batch results JSONL file
        output_dir: Directory where to save the organized results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    total_files = 0
    video_clips = set()

    with open(batch_results_file, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                # Parse the JSON line
                batch_entry = json.loads(line)

                # Extract custom_id which contains our video name and frame info
                custom_id = batch_entry.get('custom_id', '')
                if not custom_id:
                    print(f"Warning: No custom_id found in line {line_number}. Skipping.")
                    continue

                # The custom_id format should be "video_name_frame_XXXX"
                try:
                    # Split at the last underscore before "frame_"
                    parts = custom_id.split('_frame_')
                    if len(parts) != 2:
                        # If we can't split on '_frame_', try another approach
                        # Try to split at the last underscore
                        video_name = '_'.join(custom_id.split('_')[:-1])
                        frame_id = custom_id.split('_')[-1]
                    else:
                        video_name = parts[0]
                        frame_id = "frame_" + parts[1]

                except Exception as e:
                    print(
                        f"Warning: Failed to parse custom_id '{custom_id}' in line {line_number}: {e}. Using entire custom_id as video name.")
                    video_name = custom_id
                    frame_id = f"frame_{line_number:04d}"

                # Create directory for this video if it doesn't exist
                video_dir = os.path.join(output_dir, video_name)
                os.makedirs(video_dir, exist_ok=True)
                video_clips.add(video_name)

                # Create the output file path
                # Ensure frame_id has .json extension
                if not frame_id.endswith('.json'):
                    output_file = os.path.join(video_dir, f"{frame_id}.json")
                else:
                    output_file = os.path.join(video_dir, frame_id)

                # Write the response to a JSON file
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    # Write the entire batch entry as JSON
                    json.dump(batch_entry, out_f, indent=2)

                total_files += 1

            except json.JSONDecodeError:
                print(f"Warning: Failed to parse JSON in line {line_number}. Skipping.")
            except Exception as e:
                print(f"Error processing line {line_number}: {e}")

    print(f"Successfully split {total_files} files across {len(video_clips)} video clips.")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Split a batch result JSONL file into individual JSON files organized by video clip")
    parser.add_argument('batch_file', help="Path to the batch results JSONL file")
    parser.add_argument('output_dir', help="Directory where to save the organized results")
    args = parser.parse_args()

    if not os.path.isfile(args.batch_file):
        print(f"Error: Batch file not found: {args.batch_file}")
        sys.exit(1)

    split_batch_results(args.batch_file, args.output_dir)


if __name__ == '__main__':
    main()
