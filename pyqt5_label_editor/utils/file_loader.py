def load_image(image_path):
    from PyQt5.QtGui import QImage
    image = QImage(image_path)
    if image.isNull():
        raise ValueError(f"Could not load image from {image_path}")
    return image

def load_label(label_path):
    from PIL import Image
    label = Image.open(label_path)
    return label

def load_data(image_folder, label_folder):
    import os
    images = []
    labels = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(label_folder, filename.replace('.png', '.lbl').replace('.jpg', '.lbl').replace('.jpeg', '.lbl'))
            
            if os.path.exists(label_path):
                images.append(load_image(image_path))
                labels.append(load_label(label_path))
            else:
                raise FileNotFoundError(f"Label file not found for {filename}")
    
    return images, labels