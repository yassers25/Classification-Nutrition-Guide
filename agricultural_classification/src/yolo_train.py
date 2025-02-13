from ultralytics import YOLO
import os
import shutil
import yaml
import cv2
import numpy as np
from PIL import Image

def prepare_yolo_dataset(train_path, test_path, validation_path, output_path):
    os.makedirs(f"{output_path}/images/train", exist_ok=True)
    os.makedirs(f"{output_path}/images/val", exist_ok=True)
    os.makedirs(f"{output_path}/images/test", exist_ok=True)
    os.makedirs(f"{output_path}/labels/train", exist_ok=True)
    os.makedirs(f"{output_path}/labels/val", exist_ok=True)
    os.makedirs(f"{output_path}/labels/test", exist_ok=True)
    
    classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    
    def process_folder(src_path, img_dest, label_dest):
        for cls_idx, cls in enumerate(classes):
            cls_path = os.path.join(src_path, cls)
            if not os.path.isdir(cls_path):
                continue
                
            for img in os.listdir(cls_path):
                if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                # Lire l'image pour obtenir ses dimensions
                img_path = os.path.join(cls_path, img)
                image = Image.open(img_path)
                width, height = image.size
                
                # Copier l'image
                shutil.copy(img_path, os.path.join(img_dest, f"{cls}_{img}"))
                
                # Créer annotation YOLO : x_center y_center width height
                # Utiliser une boîte plus petite (80% de l'image)
                rel_width = 0.8
                rel_height = 0.8
                
                # Centre de l'image
                x_center = 0.5
                y_center = 0.5
                
                # Écrire l'annotation
                with open(os.path.join(label_dest, f"{cls}_{img.rsplit('.', 1)[0]}.txt"), 'w') as f:
                    f.write(f"{cls_idx} {x_center} {y_center} {rel_width} {rel_height}")
    
    process_folder(train_path, f"{output_path}/images/train", f"{output_path}/labels/train")
    process_folder(validation_path, f"{output_path}/images/val", f"{output_path}/labels/val")
    process_folder(test_path, f"{output_path}/images/test", f"{output_path}/labels/test")
    
    data = {
        'path': os.path.abspath(output_path),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': classes
    }
    
    with open(f"{output_path}/data.yaml", 'w') as f:
        yaml.dump(data, f)

def train_yolo():
    model = YOLO('yolov8n.pt')
    model.train(
        data='yolo_dataset/data.yaml',
        epochs=20,
        imgsz=640,
        batch=8,
        patience=5,
        name='fruits_vegetables',
        save_period=5,
        device='cpu',
        workers=4,
        mosaic=1.0,        # Augmentation des données pour améliorer la détection multiple
        mixup=0.1,         # Mixup augmentation
        copy_paste=0.1,    # Copy-paste augmentation
    )

if __name__ == "__main__":
    prepare_yolo_dataset("train", "test", "validation", "yolo_dataset")
    train_yolo()