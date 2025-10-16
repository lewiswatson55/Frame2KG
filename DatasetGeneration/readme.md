# Frame2KG-YC2 Synthetic Dataset Generation

A codebase for setting up batching jobs for transforming images into structured knowledge graphs using OpenAI's vision models with cost-efficient batch processing.

## Overview

This project provides utilities to:
1. Generate batch requests from a collection of images
2. Send these batch requests to OpenAI's API
3. Split and organise the results 
4. Validate the quality of the generated knowledge graphs

The system extracts structured knowledge from images, representing entities (nodes) and their relationships (edges) in a standardised graph format.

## Key Features

- Convert JPEG images to base64-encoded format suitable for API submission
- Batch processing to efficiently handle large numbers of images
- Custom JSON schema validation for ensuring the quality of generated graphs
- Organized output structure for easy integration with downstream applications
- Support for processing video frames with appropriate naming conventions

## Components

### Scripts

- `batch_generator.py`: Creates a JSONL file containing batch requests for image-to-graph structured outputs
- `create_large_batch.py`: Generates a large batch JSONL file from keyframes of videos across multiple directories
- `send_batch_to_OpenAI.py`: Uploads and creates an OpenAI batch job from a JSONL file
- `split_batch_result.py`: Splits a batch result JSONL file into individual JSON files organized by video clip
- `validate_batch_result.py`: Validates batch results by checking for empty or invalid content

## Usage

### 1. Generate Batch Requests

```bash
python batch)generator.py -i /path/to/images -o batch_requests.jsonl
```

Options:
- `-i, --input_folder`: Path to the folder containing image files (required)
- `-o, --output_file`: Output JSONL file (default: batch_requests.jsonl)

### 2. Create Large Batch from Multiple Video Directories

```bash
python create_large_batch.py /Volumes/M1EXT2/YouCookFrames/training/104 -o ./batches/batch-104.jsonl
# or
python create_large_batch.py /Volumes/M1EXT2/YouCookFrames/training/104 /Volumes/M1EXT2/YouCookFrames/training/105 -o ./batches/batch-104-105.jsonl
```

Options:
- `input_dirs`: List of directories containing video clip subdirectories (required)
- `-o, --output`: Output JSONL file path (required)

This script processes multiple directories containing video clips, where each video clip folder has a `_keyframes` subdirectory with JPEG images. It creates a consolidated batch file with custom IDs that preserve video and frame information.

### 3. Send Batch to OpenAI

```bash
python sendBatchToOpenAI.py [-i INPUT_FILE] [-o OUTPUT_DIR]
```

Options:
- `-i, --input_file`: Path to your .jsonl file (default: large_batch.jsonl)
- `-o, --output_dir`: Directory where to save the batch ID file (default: current directory)

### 4. Split Batch Results

```bash
python splitBatchResult.py batch_results.jsonl output_directory
```

This organizes results into a folder structure:
```
output_directory/
    video_clip_name_1/
        frame_0001.json
        frame_0002.json
        ...
    video_clip_name_2/
        frame_0001.json
        ...
```

### 5. Validate Batch Results

```bash
python validateBatchResult.py /Volumes/M1EXT2/SyntheticYouCookGraphs/Training/101/batch_6823b7af82108190a97ac08e7430eb32_output.jsonl
```

This checks for empty content fields or invalid JSON in the batch results.

## Knowledge Graph Format

The generated knowledge graphs follow this JSON schema:

```json
{
  "graph": {
    "nodes": [
      {
        "id": "person1",
        "label": "Woman",
        "attributes": {
          "appearance": "red dress",
          "size": "medium"
        },
        "location": "0.25,0.0,0.50,0.40,0.7"
      },
      ...
    ],
    "edges": [
      {
        "source": "person1",
        "target": "object1",
        "predicate": "holding"
      },
      ...
    ]
  }
}
```

## Requirements

- Python 3.6+
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)
- Required Python packages:
  - openai
  - python-dotenv

## Environment Setup

Create a `.env` file in the project root with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Notes

- Batch files include the base64 encoding of images, so they are too large for GitHub
- The system is designed to work with JPEG images
- Custom IDs are used to maintain image identity throughout the pipeline
