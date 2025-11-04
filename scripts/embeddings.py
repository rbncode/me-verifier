import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

CROPPED_DIR_ME = 'data/cropped/me'
CROPPED_DIR_NOT_ME = 'data/cropped/not_me'
OUTPUT_EMBEDDINGS_FILE = 'data/embeddings.npy'
OUTPUT_LABELS_FILE = 'data/labels.csv'

def get_image_paths(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

def generate_embeddings(image_paths, model, device):
    embeddings = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model(img_tensor)
            embeddings.append(embedding.cpu().numpy().flatten())
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return np.array(embeddings)

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    me_paths = get_image_paths(CROPPED_DIR_ME)
    not_me_paths = get_image_paths(CROPPED_DIR_NOT_ME)

    me_labels = [1] * len(me_paths)
    not_me_labels = [0] * len(not_me_paths)

    all_paths = me_paths + not_me_paths
    all_labels = me_labels + not_me_labels

    if not all_paths:
        print("No cropped images found. Please run scripts/crop_faces.py first.")
    else:
        print("Generating embeddings for all images...")
        all_embeddings = generate_embeddings(all_paths, resnet, device)

        np.save(OUTPUT_EMBEDDINGS_FILE, all_embeddings)

        df_labels = pd.DataFrame({'path': all_paths, 'label': all_labels})
        df_labels.to_csv(OUTPUT_LABELS_FILE, index=False)

        print(f"\nEmbeddings saved to {OUTPUT_EMBEDDINGS_FILE}")
        print(f"Labels saved to {OUTPUT_LABELS_FILE}")
        print(f"Total embeddings generated: {len(all_embeddings)}")

    print("\nEmbedding generation process complete.")
