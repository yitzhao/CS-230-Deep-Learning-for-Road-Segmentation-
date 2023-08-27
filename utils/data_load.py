import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
from Dataloader import RoadsDataset
# def preprocess_mask_image2(image, class_num, color_limit):
#   pic = np.array(image)
#   img = np.zeros((pic.shape[0], pic.shape[1], 1))  
#   np.place(img[ :, :, 0], pic[ :, :, 0] >= color_limit, 1)  
#   return img

# def dice_coef(y_true, y_pred):

#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0) 
 
# def dice_coef_loss(y_true, y_pred):
#     return 1-dice_coef(y_true, y_pred)

# DATA_DIR = '../input/massachusetts-roads-dataset/tiff/' # For Local
DATA_DIR = 'input/massachusetts-roads-dataset/tiff/' # For VM

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_labels')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'test_labels')

def load_datasets(visualize_image = False):
    # class_dict = pd.read_csv("../input/massachusetts-roads-dataset/label_class_dict.csv") # For Local
    class_dict = pd.read_csv("input/massachusetts-roads-dataset/label_class_dict.csv") # For VM
    # Get class names
    class_names = class_dict['name'].tolist()
    # Get class RGB values
    class_rgb_values = class_dict[['r','g','b']].values.tolist()

    print('All dataset classes and their corresponding RGB values in labels:')
    print('Class Names: ', class_names)
    print('Class RGB values: ', class_rgb_values)

    # Useful to shortlist specific classes in datasets with large number of classes
    select_classes = ['background', 'road']

    # Get RGB values of required classes
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

    print('Selected classes and their corresponding RGB values in labels:')
    print('Class Names: ', class_names)
    print('Class RGB values: ', class_rgb_values)


    dataset = RoadsDataset(x_train_dir, y_train_dir, class_rgb_values=select_class_rgb_values)
    print(len(dataset))
    random_idx = random.randint(0, len(dataset)-1)
    image, mask = dataset[2]

    if visualize_image:
        visualize(
            original_image = image,
            ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
            one_hot_encoded_mask = reverse_one_hot(mask)
        )

    #Visualize Sample Image and Mask 
    dataset = RoadsDataset(x_train_dir, y_train_dir, class_rgb_values=select_class_rgb_values)
    random_idx = random.randint(0, len(dataset)-1)
    image, mask = dataset[2]

    if visualize_image:
        visualize(
            original_image = image,
            ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
            one_hot_encoded_mask = reverse_one_hot(mask)
        )
    
    #Visualize Augmented Images & Masks
    augmented_dataset = RoadsDataset(
        x_train_dir, y_train_dir, 
        augmentation=get_training_augmentation(),
        class_rgb_values=select_class_rgb_values,
    )

    random_idx = random.randint(0, len(augmented_dataset)-1)

    # Different augmentations on a random image/mask pair (256*256 crop)
    for i in range(3):
        image, mask = augmented_dataset[random_idx]
        if visualize_image:
            visualize(
                original_image = image,
                ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
                one_hot_encoded_mask = reverse_one_hot(mask)
            )

    train_dataset = RoadsDataset(
    x_train_dir, y_train_dir, 
    augmentation=get_training_augmentation(),
    # preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=select_class_rgb_values,
    )

    valid_dataset = RoadsDataset(
        x_valid_dir, y_valid_dir, 
        augmentation=get_training_augmentation(), 
        # preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )
    test_dataset = RoadsDataset(
        x_test_dir, y_test_dir, 
        augmentation=get_training_augmentation(), 
        # preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    # Get train and val data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    return train_loader, valid_loader, test_loader
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map
    
# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

def get_training_augmentation():
    train_transform = [    
        album.RandomCrop(height=256, width=256, always_apply=True),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():   
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)