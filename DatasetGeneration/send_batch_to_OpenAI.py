#!/usr/bin/env python3
"""
Upload and create an OpenAI batch job from a JSONL file in the bigbatch directory.
Defaults to 'large_batch.jsonl' and writes batch ID files into the same folder.
Usage:
    python sendBatchToOpenAI.py [-i INPUT_FILE] [-o OUTPUT_DIR]
"""
import os
import sys
import datetime
import argparse
from openai import OpenAI
from dotenv import load_dotenv

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# First try to load .env from script directory, then from project root if not found
dotenv_path = os.path.join(script_dir, '.env')
if not os.path.exists(dotenv_path):
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    dotenv_path = os.path.join(project_root, '.env')

load_dotenv(dotenv_path, override=True)

# Validate API key
if not os.getenv("OPENAI_API_KEY", "").endswith("Gux4A"):
    raise Exception("OpenAI SyntheticDataset API Project key environment variable not set.")
else:
    print("OpenAI SyntheticDataset API Project key loaded successfully.")

# Initialise client
client = OpenAI()

def upload_batch_file(jsonl_filepath: str) -> str:
    """Uploads the JSONL file to OpenAI and returns the file ID."""
    with open(jsonl_filepath, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    print(f"Uploaded file ID: {uploaded_file.id}")
    return uploaded_file.id


def create_batch_job(file_id: str):
    """Creates a batch job and returns the batch object."""
    batch_job = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Batch created. ID: {batch_job.id}")
    return batch_job


def main():
    parser = argparse.ArgumentParser(
        description="Upload a JSONL batch and create an OpenAI batch job."
    )
    parser.add_argument(
        "-i", "--input_file",
        default=os.path.join(script_dir, "large_batch.jsonl"),
        help="Path to your .jsonl file (default: large_batch.jsonl in this folder)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        default=script_dir,
        help="Directory where to save the batch ID file (default: this folder)."
    )
    args = parser.parse_args()

    input_file = args.input_file
    if not os.path.isfile(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    file_id = upload_batch_file(input_file)
    batch_job = create_batch_job(file_id)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"batch_id_{timestamp}.txt")
    with open(output_path, "w") as f:
        f.write(batch_job.id)
    print(f"Saved batch ID to {output_path}")


if __name__ == "__main__":
    main()
