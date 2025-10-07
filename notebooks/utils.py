# %load utils.py

from torchvision import transforms
import torchvision
from PIL import Image

def load_image(image_path):
    # image = Image.open(image_path)
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = transforms.ToTensor()(image)
    # print(image.shape)
    # image = image.float() / 255
    preprocess = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = preprocess(image)
    # print(image.max(), image.min())
    return image

def forward_gen(x_0, eg, el):
    model.netG(x_0, eg, el, trainer.pos_emb)
