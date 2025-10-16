#!/usr/bin/env python3
"""
Script to generate a large batch JSONL file from keyframes of videos across multiple directories.
Usage:
    python create_large_batch.py <input_dirs>... -o output.jsonl
"""
import os
import sys
import json
import argparse

# ensure newBatchGenerator is importable
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from synthetic_dataset_batch_generator import encode_image_to_base64, create_batch_request

def generate_large_batch(input_dirs, output_file):
    requests = []
    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            print(f"Warning: {input_dir} is not a directory. Skipping.")
            continue
        # iterate through each video clip folder
        for video_name in sorted([d for d in os.listdir(input_dir)
                                  if os.path.isdir(os.path.join(input_dir, d))]):
            keyframe_dir = os.path.join(input_dir, video_name, '_keyframes')
            if not os.path.isdir(keyframe_dir):
                print(f"Warning: Keyframe directory not found for video {video_name} in {input_dir}. Skipping.")
                continue
            # list .jpg keyframe files
            image_files = sorted([f for f in os.listdir(keyframe_dir)
                                  if f.lower().endswith('.jpg') and not f.startswith('._')])
            if not image_files:
                print(f"Warning: No JPEG images in {keyframe_dir}. Skipping.")
                continue
            for img in image_files:
                image_path = os.path.join(keyframe_dir, img)
                image_b64 = encode_image_to_base64(image_path)
                frame_id = os.path.splitext(img)[0]
                # include video folder name and frame number in custom_id
                custom_id = f"{video_name}_{frame_id}"
                req = create_batch_request(image_b64, custom_id)
                requests.append(req)
    # write all requests to JSONL
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for req in requests:
            out_f.write(json.dumps(req) + '\n')
    print(f"Generated {len(requests)} batch requests in {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate large batch JSONL from video keyframes.")
    parser.add_argument('input_dirs', nargs='+', help="List of directories containing video clip subdirectories.")
    parser.add_argument('-o', '--output', required=True, help="Output JSONL file path.")
    args = parser.parse_args()
    generate_large_batch(args.input_dirs, args.output)

if __name__ == '__main__':
    main()
