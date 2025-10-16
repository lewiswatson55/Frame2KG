import os
import shutil
from PIL import Image
import torch
from torchvision import transforms
import open_clip
from tqdm import tqdm
from torch.nn.functional import cosine_similarity

# add a 1 hour waiting time bc i want to stack them and still sleep tonight
import time
time.sleep(3600)

# === CONFIGURATION ===
parent_video_folder = "/Volumes/M1EXT2/YouCookFrames/testing/"
threshold = 0.90  # Bigger = less frames kept
list_of_averages = []

# === Load Model ===
device = "mps" #if torch.backends.mps.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to(device)
model.eval()

def get_embedding(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(image).squeeze().cpu()

def filter_keyframes_in_folder(folder_path, threshold=0.90):
    """
    If keyframes already exist in the '_keyframes' subfolder:
        - Identify the last keyframe file (sorted by name).
        - Find it in the main folder's sorted frames.
        - Continue threshold-checking from there onward.
    """
    # --- Gather frames in the main folder ---
    frame_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    if not frame_files:
        return  # No images, so do nothing

    # --- Prepare the _keyframes subfolder ---
    keyframe_folder = os.path.join(folder_path, "_keyframes")
    os.makedirs(keyframe_folder, exist_ok=True)

    # --- Gather frames already in _keyframes ---
    existing_keyframes = sorted([
        f for f in os.listdir(keyframe_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    # --- Determine where to resume from ---
    if existing_keyframes:
        # Pick the last keyframe in alphabetical order
        last_keyframe_file = existing_keyframes[-1]
        # If that file also exists in the main folder, use it as our "prev_embedding"
        if last_keyframe_file in frame_files:
            last_index = frame_files.index(last_keyframe_file)
            prev_embedding = get_embedding(os.path.join(folder_path, last_keyframe_file))
            start_index = last_index + 1
        else:
            # If it's not in the main folder for some reason, just treat as fresh
            prev_embedding = get_embedding(os.path.join(folder_path, frame_files[0]))
            start_index = 1
    else:
        # No keyframes yet, so copy the very first frame
        first_frame_path = os.path.join(folder_path, frame_files[0])
        shutil.copy2(first_frame_path, os.path.join(keyframe_folder, frame_files[0]))
        prev_embedding = get_embedding(first_frame_path)
        start_index = 1

    # --- Continue from the chosen start index ---
    for i in range(start_index, len(frame_files)):
        frame_name = frame_files[i]
        full_frame_path = os.path.join(folder_path, frame_name)

        # Skip if already in _keyframes
        if frame_name in existing_keyframes:
            continue

        # Compute similarity with the last chosen keyframe
        current_embedding = get_embedding(full_frame_path)
        sim = cosine_similarity(prev_embedding, current_embedding, dim=0).item()

        if sim < threshold:
            shutil.copy2(full_frame_path, os.path.join(keyframe_folder, frame_name))
            prev_embedding = current_embedding
            existing_keyframes.append(frame_name)  # Mark as now included

    # --- Print how many frames are in _keyframes ---
    final_count = len([
        f for f in os.listdir(keyframe_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"✓ {os.path.basename(folder_path)}: now has {final_count} keyframes out of {len(frame_files)} frames")
    list_of_averages.append(final_count / len(frame_files))

def process_all_videos(parent_folder, threshold=0.90):
    """
    Recursively finds all folders containing images and runs filter_keyframes_in_folder on each.
    """
    folders_with_images = []
    for root, dirs, files in os.walk(parent_folder):
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            folders_with_images.append(root)

    folders_with_images = sorted(folders_with_images)  # stable order for the progress bar

    for folder_path in tqdm(folders_with_images, desc="Processing all folders"):
        filter_keyframes_in_folder(folder_path, threshold)

    # Print overall stats
    average = calculate_average(list_of_averages)
    print(f"\nAverage frames kept per folder: {average:.2f}")

def calculate_average(list_of_averages):
    if not list_of_averages:
        return 0
    return sum(list_of_averages) / len(list_of_averages)

if __name__ == "__main__":
    process_all_videos(parent_video_folder, threshold=threshold)
