# Implementation based on https://github.com/PAIR-code/saliency from XRAI: Better Attributions Through Regions
# Install necessary packages
!pip install saliency torch torchvision

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm_notebook
from io import StringIO
from scipy import ndimage
from textwrap import wrap
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab as P
import PIL.Image
import io
import saliency.core as saliency

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load and modify ResNet18 model
adv_model = resnet18()
adv_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
adv_model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
adv_model.fc = nn.Linear(512, 10)

adv_model.load_state_dict(torch.load("/content/drive/MyDrive/adv_cifar10_model.pth")["model_state_dict"])
adv_pruned_model = torch.load('/content/drive/MyDrive/adv_cifar10_model_pruned.pth')['model']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
adv_model.to(device)
adv_pruned_model.to(device)

# Switching to evaluation mode
adv_model.eval()
adv_pruned_model.eval()

# Helper functions for displaying images
def ShowImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im)
    P.title(title)

def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)

def ShowHeatMap(im, title, ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap='inferno')
    P.title(title)

# Normalize images
normalizer = transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))

def PreprocessImages(images):
    images = np.array(images) / 255
    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.tensor(images, dtype=torch.float32)
    images = normalizer(images)
    return images.requires_grad_(True)

# Register hooks for Grad-CAM
conv_layer = adv_model.layer4[-1].conv2
conv_layer_outputs = {}

def conv_layer_forward(m, i, o):
    conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().cpu().numpy()

def conv_layer_backward(m, i, o):
    conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 3).detach().cpu().numpy()

conv_layer.register_forward_hook(conv_layer_forward)
conv_layer.register_full_backward_hook(conv_layer_backward)

# Define call_model_function for saliency
class_idx_str = 'class_idx_str'

# Wrapper function for the adversarial model
def call_model_function(images, call_model_args=None, expected_keys=None):
    images = PreprocessImages(images)
    target_class_idx = call_model_args[class_idx_str]
    output = adv_model(images)
    m = torch.nn.Softmax(dim=1)
    output = m(output)
    
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        outputs = output[:, target_class_idx]
        grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
        grads = torch.movedim(grads[0], 1, 3)
        gradients = grads.detach().numpy()
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
    else:
        one_hot = torch.zeros_like(output)
        one_hot[:, target_class_idx] = 1
        adv_model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        return conv_layer_outputs

# Load and preprocess CIFAR-10 dataset
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_set, batch_size=200, shuffle=False)
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Get a batch of test images
images, labels = next(iter(test_loader))

# Load a single image for visualization
im_orig = images[191].cpu().numpy().transpose(1, 2, 0)
im_tensor = PreprocessImages([im_orig]).to(device)

# Show original image
ShowImage(im_orig)
predictions = adv_model(im_tensor).detach().cpu().numpy()
prediction_class = np.argmax(predictions[0])
call_model_args = {class_idx_str: prediction_class}

print("Prediction class: " + str(prediction_class))  # Should be a doberman, class idx = 236

# Compute XRAI attributions
xrai_object = saliency.XRAI()
adv_model.to('cpu')
xrai_attributions = xrai_object.GetMask(im_orig.astype(np.float32), call_model_function, call_model_args, batch_size=20)

# Set up matplotlib figures
ROWS = 1
COLS = 3
UPSCALE_FACTOR = 20
P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# Show original image
ShowImage(im_orig, title='Original Image', ax=P.subplot(ROWS, COLS, 1))

# Show XRAI heatmap attributions
ShowHeatMap(xrai_attributions, title='XRAI Heatmap', ax=P.subplot(ROWS, COLS, 2))

# Show most salient 30% of the image
mask = xrai_attributions >= np.percentile(xrai_attributions, 70)
im_mask = np.array(im_orig)
im_mask[~mask] = 0
ShowImage(im_mask, title='Top 30%', ax=P.subplot(ROWS, COLS, 3))
