#!/usr/bin/env python3
"""
This script generates a JSONL file containing batch requests for image-to-graph structured outputs.
It reads JPEG images from a specified folder, encodes them to base64, and creates a JSON object for each image.

Each frame is given a customID in the batch.

Hyperparameters:
  - model: "gpt-4o"
  - detail: "high"
  - max_tokens: 1500

This code does not call the OpenAI API – it just assembles the JSONL file.
"""

import os
import json
import base64
import argparse

# Hyperparameters
model = "o4"
detail = "high"
max_tokens = 30000
reasoning_effort = "medium"
store = True

# System prompt – the text that goes in the system message.
SYSTEM_PROMPT = """
You are a vision–language model that must output a single-line JSON object called **KnowledgeGraph**.

Hard rules  
1. Do **not** include code-blocks, explanations, or extra keys—only the JSON object.  
2. The JSON must pass the attached JSON-Schema **without whitespace, tabs, or line-breaks**.  
3. If an attributes field would be empty, write **null** (never {}).

Required content  
A. **Nodes** — one per distinct entity you can see. Each node object **must** contain:  
   • "id": short unique ID, e.g., "person1"  
   • "label": concise noun phrase, e.g., "Electric Guitar"  
   • "attributes": object or null. When possible, include **at least two** of:  
     - "appearance": dominant visual description, e.g., "red cotton", "blue plastic".  
     - "size": relative size in the scene: "small", "medium", "large".
    • "location": normalised bounding box plus confidence e.g. "0.25,0.0,0.50,0.40,0.7"

B. **Edges** — one per relationship you identify. Each edge object **must** contain:  
   • "source": node ID  
   • "target": node ID  
   • "predicate": pick from the list below **or**, if none fit, choose a single clear verb/preposition (snake_case) that best describes the relation.

["holding","wearing","sitting_on","standing_on","next_to","left_of","right_of","behind","in_front_of","attached_to","supports","connected_to","includes","has_component","on","in"]

Quality goals  
• Capture **every meaningful object** in foreground **and** background.  
• Use the richest applicable attributes—**appearance** and **location** are high-value.  
• Most images contain many objects; returning only one or two is almost certainly wrong.

Return the JSON in **one line** with no spaces or line-breaks."""

# Updated JSON Schema for the KnowledgeGraph response.
# Note: Both nodes and edges now include "attributes" in their "required" arrays.
JSON_SCHEMA = {
  "name": "KnowledgeGraph",
  "strict": True,
  "schema": {
    "type": "object",
    "required": [
      "graph"
    ],
    "properties": {
      "graph": {
        "type": "object",
        "required": [
          "nodes",
          "edges"
        ],
        "properties": {
          "edges": {
            "type": "array",
            "items": {
              "type": "object",
              "required": [
                "source",
                "target",
                "predicate"
              ],
              "properties": {
                "source": {
                  "type": "string"
                },
                "target": {
                  "type": "string"
                },
                "predicate": {
                  "type": "string"
                }
              },
              "additionalProperties": False
            }
          },
          "nodes": {
            "type": "array",
            "items": {
              "type": "object",
              "required": [
                "id",
                "label",
                "attributes",
                "location"
              ],
              "properties": {
                "id": {
                  "type": "string"
                },
                "label": {
                  "type": "string"
                },
                "location": {
                  "type": [
                    "string"
                  ]
                },
                "attributes": {
                  "type": "object",
                  "required": [
                    "appearance",
                    "size"
                  ],
                  "properties": {
                    "size": {
                      "type": [
                        "string",
                        "null"
                      ]
                    },
                    "appearance": {
                      "type": [
                        "string",
                        "null"
                      ]
                    }
                  },
                  "additionalProperties": False
                }
              },
              "additionalProperties": False
            }
          }
        },
        "additionalProperties": False
      }
    },
    "additionalProperties": False
  }
}



def encode_image_to_base64(image_path: str) -> str:
    """Reads and encodes an image file to a base64 string."""
    with open(image_path, "rb") as f:
        encoded_bytes = base64.b64encode(f.read())
    return encoded_bytes.decode("utf-8")

def create_batch_request(image_b64: str, custom_id: str) -> dict:
    """Creates the JSON object for a single batch request using the image data."""
    request_obj = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + image_b64,
                            }
                        }
                    ]
                }
            ],
            "max_completion_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "store": store,
            "response_format": {
                "type": "json_schema",
                "json_schema": JSON_SCHEMA
            }
        }
    }
    return request_obj

def create_batch_jsonl(image_folder: str, output_file: str):
    """
    Reads JPEG images from the given folder and writes a JSONL file
    where each line is a batch request for generating an image-graph pair.
    """
    # Get list of files that end with .jpg (case-insensitive) and sort them.
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(".jpg")
    ])
    if not image_files:
        print("No JPG images found in the folder.")
        return

    with open(output_file, "w", encoding="utf-8") as out_f:
        for filename in image_files:
            image_path = os.path.join(image_folder, filename)
            b64_img = encode_image_to_base64(image_path)
            # Remove file extension for custom_id (e.g. "frame_0001")
            custom_id = os.path.splitext(filename)[0]
            batch_request = create_batch_request(b64_img, custom_id)
            # Write one JSON object per line.
            out_f.write(json.dumps(batch_request) + "\n")
            print(f"Added batch request for {filename} as {custom_id}")

def main():
    parser = argparse.ArgumentParser(
        description="Create a batch JSONL file for image-to-graph structured outputs."
    )
    parser.add_argument(
        "-i", "--input_folder",
        required=True,
        help="Path to the folder containing image files (e.g. frames)"
    )
    parser.add_argument(
        "-o", "--output_file",
        default="batch_requests.jsonl",
        help="Output JSONL file (default: batch_requests.jsonl)"
    )
    args = parser.parse_args()
    create_batch_jsonl(args.input_folder, args.output_file)
    print(f"Batch JSONL file created: {args.output_file}")

if __name__ == "__main__":
    main()
