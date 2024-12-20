import os
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

from torchvision.transforms import RandomHorizontalFlip, RandomRotation
from PIL import ImageEnhance

def augment_image(image):
    image = RandomHorizontalFlip()(image)
    image = RandomRotation(10)(image)
    image = ImageEnhance.Contrast(image).enhance(1.2)
    return image


def load_data(dataset_path, image_size=(64, 64)):
    data = []
    labels = []
    class_names = sorted(os.listdir(dataset_path))
    
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            try:
                image = Image.open(image_path).convert('RGB')
                image = image.resize(image_size)
                data.append(np.array(image))
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

    data = np.array(data)
    labels = np.array(labels)
    return train_test_split(data, labels, test_size=0.2, random_state=42), class_names