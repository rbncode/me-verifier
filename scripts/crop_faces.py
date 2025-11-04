import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch

INPUT_DIR_ME = 'data/me'
INPUT_DIR_NOT_ME = 'data/not_me'
OUTPUT_DIR_ME = 'data/cropped/me'
OUTPUT_DIR_NOT_ME = 'data/cropped/not_me'
IMAGE_SIZE = 160

def process_and_crop(input_dir, output_dir, mtcnn):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            img = Image.open(input_path).convert('RGB')

            boxes, _ = mtcnn.detect(img)

            if boxes is not None:
                box = boxes[0]

                face = img.crop(box)
                face = face.resize((IMAGE_SIZE, IMAGE_SIZE))
                face.save(output_path)
                print(f"Cropped face saved to {output_path}")
            else:
                print(f"No face detected in {input_path}")

        except Exception as e:
            print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    mtcnn = MTCNN(keep_all=True, device=device)

    print("Starting to process 'me' images...")
    process_and_crop(INPUT_DIR_ME, OUTPUT_DIR_ME, mtcnn)

    print("\nStarting to process 'not_me' images...")
    process_and_crop(INPUT_DIR_NOT_ME, OUTPUT_DIR_NOT_ME, mtcnn)

    print("\nFace cropping process complete.")
