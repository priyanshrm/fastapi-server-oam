import torch
import io
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image


class AutoEncoderV2(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
    )

    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(32, 32, 3, stride=2, output_padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 16, 3, stride=2, output_padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 8, 3, stride=2, output_padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=0)
      )

  def forward(self, x:torch.Tensor)->torch.Tensor:
    x = self.encoder(x)
    x = self.decoder(x)
    return x

def transformImage(image):
    mean_, std_ = [0.5,0.5,0.5], [0.5,0.5,0.5]
    transform1 = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])
    transform2 = transforms.Compose([
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])
    transform3 = transforms.Compose([
        transforms.Normalize(mean=mean_, std=std_)
    ])
    # Apply the transformation function to the image
    image = transform1(image)
    
    # Check if the image has the correct shape
    if image.shape != (3, 256, 256):
        image = transform2(image)
    image = transform3(image)
    return image

def getEncoding(image_bytes):
    pil_image = Image.open(io.BytesIO(image_bytes))
    transformed_image = transformImage(pil_image)
    model_1 = AutoEncoderV2()
    model_1.load_state_dict(torch.load("./cnn/oam_model_v2_2", map_location=torch.device('cpu')))
    model_1.eval()
    with torch.inference_mode():
        output = model_1(transformed_image)
    return output

def getStrEncoding(image):
    output = getEncoding(image)
    array = output.numpy()
    tensor_str = array.tostring()
    return tensor_str

def parseStrEncoding(tensor_str):
    array = np.frombuffer(tensor_str, dtype=np.float32)
    tensor = torch.from_numpy(array.reshape(3, 256, 256))
    return tensor

def cosine_similarity(strEncoding1, strEncoding2):
    img1, img2 = parseStrEncoding(strEncoding1), parseStrEncoding(strEncoding2)
    img1_flat = torch.flatten(img1)
    img2_flat = torch.flatten(img2)
    cos_sim = F.cosine_similarity(img1_flat, img2_flat, dim=0)
    return cos_sim.item()

