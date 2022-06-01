from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def load_batch(path_dir: str, batch_size: int =64, resize: int =255, 
                                    crop_size: int =225, degree_rot: int =30)->Tuple[DataLoader]:
    
    """
    Return a tuple of DataLoader, batches which will be shuffle dataset with pytorch DataLoader.
    train_loader and test_loader: containing the training tensor transformation images pixels 
    and the labels of the dataset encoded based on the arrangement of the folders containing the images.

    Parameters:
    ----------

    path_dir: str, Required
        Is the path containing the folders where the individual images can found.

    batch_size: int, default set as (64)
        This determine the number of dataset in each batch for the training, and test data

    resize: int, default = 255

    """
    
    train_transforms = transforms.Compose([transforms.Resize(resize),
                                        transforms.RandomRotation(degree_rot),
                                        transforms.RandomResizedCrop(crop_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5])])

    test_transforms = transforms.Compose([transforms.Resize(resize),
                                        transforms.CenterCrop(crop_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])

    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(path_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(path_dir , transform=test_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    class_list=train_data.classes
    return train_loader, test_loader, class_list


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax



def view_classify_general(img, ps, class_list):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    imshow(img, ax=ax1, normalize=True)
    ax1.axis('off')
    ax2.barh(np.arange(len(class_list)), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(class_list)))
    ax2.set_yticklabels([x for x in class_list], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
