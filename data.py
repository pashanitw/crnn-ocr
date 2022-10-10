##
import torch
from PIL import Image
from torch.utils.data import DataLoader
import torchdata.datapipes as dp
from pathlib import Path
from torchvision import transforms
import os
from torchtext import vocab
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

img_width = 200
img_height = 50


##
def getDataInfo():
    data_dir = Path("./captcha_images_v2/")

    # Get list of all the images
    images = sorted(list(map(str, list(data_dir.glob("*.png")))))
    labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
    characters = set(char for label in labels for char in label)
    characters = sorted(list(characters))

    print("Number of images found: ", len(images))
    print("Number of labels found: ", len(labels))
    print("Number of unique characters: ", len(characters))
    print("Characters present: ", characters)

    # Batch size for training and validation
    batch_size = 16

    # Desired image dimensions
    img_width = 200
    img_height = 50

    # Factor by which the image is going to be downsampled
    # by the convolutional blocks. We will be using two
    # convolution blocks and each block will have
    # a pooling layer which downsample the features by a factor of 2.
    # Hence total downsampling factor would be 4.
    downsample_factor = 4

    # Maximum length of any captcha in the dataset
    max_length = max([len(label) for label in labels])
    return max_length,characters, downsample_factor
##
def get_lookup():
    return vocab.vocab(Counter(characters))
##
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
         #   transforms.RandomCrop((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            # normalize images to [-1, 1] range
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
}
##
def img_path_to_label(path: str):
    """Function to get the class from the file path"""
    label = Path(path).stem
    lookup = get_lookup()
    return path, lookup.lookup_indices(list(label))
##
def filter_png(x):
    return x.endswith('.png')
##

def open_image(x):
    img = Image.open(x[0])
    return Image.open(x[0]), torch.tensor(x[1], dtype=torch.int64)
##
def apply_transforms(x):
    return data_transforms['train'](x[0]), x[1]
##
def build_datapipes(path):
    new_dp = dp.iter.FileLister(path, recursive=False)
    print("==== coming here =====")
    new_dp = new_dp.filter(filter_png)
    new_dp = new_dp.map(img_path_to_label).enumerate()
    return  new_dp\
        .to_map_datapipe()\
        .map(open_image) \
        .map(apply_transforms)
##
# Iterate over the 5 first batches

##


max_length,characters, downsample_factor = getDataInfo()
##

def create_dataloaders():
    data_iter = build_datapipes('./captcha_images_v2')
    train_size = int(0.9 * len(data_iter))
    val_size = len(data_iter) - train_size
    train_dataset, val_dataset = random_split(data_iter, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    return train_loader, val_loader
##

data_iter = build_datapipes('./captcha_images_v2')
train_loader = DataLoader(data_iter, batch_size=2, shuffle=True)

for i, (data, target) in enumerate(train_loader):
    print("==== coming here =====")
    print(target)
    print("======== done ========")
    break

##
def getVocabSize():
    return len(characters)
##
