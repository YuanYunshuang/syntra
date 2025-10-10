import torch
from torchvision import transforms
from PIL import Image

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")


class Dino2Encoder:
    def __init__(self):
        # Load the DINOv2 model
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', source='github')
        self.model.cuda()  # Set the model to evaluation mode

    def generate_embedding(self, input_tensor):
        # Generate embeddings
        with torch.no_grad():  # Disable gradient computation
            embeddings = self.model.forward_features(input_tensor)

        return embeddings



if __name__ == "__main__":
    # Example usage
    image_paths = ["assets/images/siegfried_vy_1.tif", "assets/images/siegfried_vy_1.tif"]
    encoder = Dino2Encoder()
    preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),          # Convert to tensor
            transforms.Normalize(           # Normalize using ImageNet mean and std
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    for image_path in image_paths:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')  # Ensure RGB format
        input_tensor = preprocess(image).unsqueeze(0).cuda()  # Add batch dimension
        embeddings = encoder.generate_embedding(image_path)
        for k, v in embeddings.items():
            print(k, v.shape if v is not None else v)