from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        if split not in ['train', 'test']:
            raise ValueError("split must be equal to 'train' or 'test'")

        #TODO: check that root exists
        with open(split + ".txt") as f:
            file_list = f.readlines()
        #remove class "BACKGROUND_Google"
        file_list = [f for f in file_list if not f.startswith("BACKGROUND_Google")]

        self.categories = set([f.split('/')[0] for f in file_list])
        self.categories = sorted(list(self.categories))

        self.y = []
        self.index = []

        for (i, c) in enumerate(categories):
            files = [f.split('_')[1] for f in file_list if f.startswith(c)]
            n = len(files)
            index.extend(files)
            y.extend(n*[i])
        #self.split = file_list
        # This defines the split you are going to use
        # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class)
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        img_name = os.path.join(self.root, "101_ObjectCategories",
                                self.categories[self.y[index]],
                                f"image_{self.index[index]: 04d})
        image = Image.open(img_name)
        label = self.y[index]

        #image, label =

        # Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.index) # Provide a way to get the length (number of elements) of the dataset
        return length
