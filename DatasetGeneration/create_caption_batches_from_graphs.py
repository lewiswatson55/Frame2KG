#!/usr/bin/env python3
"""
Simple script to create caption generation batches from graphs in HF dataset.

Dataset structure (lewiswatson/Frame2KG-YC2):
- video_id: string
- frame_number: string (not sequential, only keyframes)
- category: string
- image: PIL Image
- graph: dict

Creates JSONL batch files for OpenAI API with:
- System prompt: "Based on the following graph representation of a frame (image), 
  generate a caption representative of the scene. Don't include anything else."
- User message: The graph as JSON string
"""

import os
import json
import argparse
from typing import Optional

try:
    from datasets import load_dataset
    from datasets.features import features as hf_features

    if not getattr(hf_features.generate_from_dict, "_handles_legacy_list", False):
        _original_generate_from_dict = hf_features.generate_from_dict

        def _patched_generate_from_dict(obj):
            if isinstance(obj, dict) and obj.get("_type") == "List":
                feature = obj.get("feature")
                return [_patched_generate_from_dict(feature)]
            return _original_generate_from_dict(obj)

        _patched_generate_from_dict._handles_legacy_list = True  # type: ignore[attr-defined]
        hf_features.generate_from_dict = _patched_generate_from_dict  # type: ignore[attr-defined]
except ImportError:
    raise SystemExit("Missing dependency: datasets. Install with `pip install datasets`")

CAPTION_SYSTEM_PROMPT = "Based on the following graph representation of a frame (image), generate a caption representative of the scene. Don't include anything else."


def create_batch_request(video_id: str, frame_number: str, graph: dict, model: str, max_tokens: int) -> dict:
    """Create a single batch request."""
    custom_id = f"{video_id}_frame_{frame_number}"
    graph_str = json.dumps(graph, separators=(",", ":"))
    
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                {"role": "user", "content": graph_str}
            ],
            "max_completion_tokens": max_tokens,
            "reasoning_effort": "low",
            "store": True
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Create caption batches from HF dataset graphs")
    parser.add_argument("--repo-id", default="lewiswatson/Frame2KG-YC2", help="HF dataset repo")
    parser.add_argument("--output-dir", default="./batches-caption", help="Output directory")
    parser.add_argument("--model", default="o4", help="Model name")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max completion tokens")
    parser.add_argument("--limit", type=int, help="Limit rows per split for testing")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading dataset: {args.repo_id}")
    # Load without streaming to avoid schema issues
    dataset = load_dataset(args.repo_id, trust_remote_code=True)
    
    splits = ["training", "validation", "testing"]
    
    for split in splits:
        if split not in dataset:
            print(f"⚠️ Split '{split}' not found")
            continue
            
        data = dataset[split]
        output_file = os.path.join(args.output_dir, f"captions_{split}.jsonl")
        
        count = 0
        max_rows = min(len(data), args.limit) if args.limit else len(data)
        
        with open(output_file, "w") as f:
            for i in range(max_rows):
                row = data[i]
                
                # Skip if no graph
                if not row.get("graph"):
                    continue
                    
                request = create_batch_request(
                    video_id=row["video_id"],
                    frame_number=row["frame_number"],
                    graph=row["graph"],
                    model=args.model,
                    max_tokens=args.max_tokens
                )
                
                f.write(json.dumps(request) + "\n")
                count += 1
        
        print(f"✅ Wrote {count} requests to {output_file}")


if __name__ == "__main__":
    main()
