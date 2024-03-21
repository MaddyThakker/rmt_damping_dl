import numpy as np
import torch
import torchvision
import os

from curvature.imagenet32_old import IMAGENET32
from torchvision.datasets import ImageFolder

class FilteredDataset(torch.utils.data.Dataset):
    """A custom dataset wrapper that filters by class indices."""
    def __init__(self, original_dataset, class_idx_need):
        self.original_dataset = original_dataset
        self.class_idx_need = class_idx_need

        self.filtered_indices = [
            i for i, (_, label) in enumerate(original_dataset)
            if label in self.class_idx_need
        ]

    def __getitem__(self, index):
        # Map the filtered index back to the original dataset's index
        orig_index = self.filtered_indices[index]
        return self.original_dataset[orig_index]

    def __len__(self):
        return len(self.filtered_indices)
    

def get_num_classes(dataset, train_set):
    if dataset == 'ImageFolder':
        return 1000  # Assuming a fixed number of classes for ImageFolder
    else:
        return len(np.unique(train_set.targets)) if isinstance(train_set.targets, torch.Tensor) else len(np.unique(train_set.targets))

def filter_by_class_indices(dataset, class_indices):
    """Filter a dataset to only include items with specified class indices."""
    # Find indices of dataset items with the desired class labels
    indices = [i for i, (_, label) in enumerate(dataset.dataset) if label in class_indices]

    # Create and return a Subset of the dataset based on these indices
    return torch.utils.data.Subset(dataset.dataset, indices)

def datasets(
        dataset,
        path,
        transform_train,
        transform_test,
        use_validation=True,
        val_size=5000,
        train_subset=None,
        train_subset_seed=None):
    print(f'Loading {dataset} from {path}')

    path = os.path.join(path, dataset.lower())
    if dataset == 'ImageNet32':
        train_set = IMAGENET32(root=path, train=True, download=False, transform=transform_train)
    elif dataset == 'ImageFolder':
        train_set = ImageFolder(root=os.path.join(path, 'train'), transform=transform_train)
    else:
        ds = getattr(torchvision.datasets, dataset)
        train_set = ds(root=path, train=True, download=True, transform=transform_train)

    num_classes = get_num_classes(dataset, train_set)

    n_train_samples = len(train_set)
    val_size = int(n_train_samples * val_size) if isinstance(val_size, float) else val_size

    # Adjusted handling for validation and test sets
    if use_validation:
        # When validation is used, create validation set from train_set
        indices = torch.randperm(len(train_set)).tolist()
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        train_set = torch.utils.data.Subset(train_set, train_indices)
        
        # For ImageFolder, validation set is from a separate folder
        if dataset == 'ImageFolder':
            val_set = ImageFolder(root=os.path.join(path, 'val'), transform=transform_test)
        else:
            # For other datasets, validation set is a subset of the training data
            val_set = torch.utils.data.Subset(train_set, val_indices)
    else:
        val_set = None  # No validation set used

    # Corrected test set creation
    if dataset == 'ImageFolder':
        test_set = ImageFolder(root=os.path.join(path, 'val'), transform=transform_test)
    else:
        # For non-ImageFolder datasets, create test set directly from the dataset
        test_set = ds(root=path, train=False, download=True, transform=transform_test)

    if train_subset is not None:
        order = torch.randperm(len(train_set)).numpy()
        if train_subset_seed is not None:
            np.random.seed(train_subset_seed)
            np.random.shuffle(order)
        train_subset_indices = order[:train_subset]
        train_set = torch.utils.data.Subset(train_set, train_subset_indices)

    print(f'Using train ({len(train_set)})' + (f', val ({len(val_set)})' if val_set else '') + f', test ({len(test_set)})')

    # Return both the validation set and the test set along with the training set
    return {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }, num_classes

def loaders(
        dataset,
        path,
        batch_size,
        num_workers,
        transform_train,
        transform_test,
        use_validation=True,
        val_size=5000,
        shuffle_train=True, 
        class_subset=None):

    ds_dict, num_classes = datasets(
        dataset, path, transform_train, transform_test, use_validation=use_validation, val_size=val_size)

    if class_subset is not None:
        # Apply filtering
        print(class_subset)
        ds_dict['train'] = FilteredDataset(ds_dict['train'], class_subset)
        ds_dict['test'] = FilteredDataset(ds_dict['test'], class_subset)
        if ds_dict.get('val'):
            ds_dict['val'] = FilteredDataset(ds_dict['val'], class_subset)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        ds_dict['train'],
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        ds_dict['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = None
    if ds_dict.get('val'):
        val_loader = torch.utils.data.DataLoader(
            ds_dict['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return {
        'train': train_loader,
        'test': test_loader,
        'val': val_loader
    }, num_classes


class CIFAR10AUG(torch.utils.data.Dataset):
    base_class = torchvision.datasets.CIFAR10

    def __init__(self, root, train=True, transform=None, download=False, shuffle_seed=1):
        self.base = self.base_class(root, train=train, transform=None, target_transform=None, download=download)
        self.transform = transform

        self.pad = 4
        self.size = len(self.base) * (2 * self.pad + 1) * (2 * self.pad + 1) * 2
        rng = np.random.RandomState(shuffle_seed)
        self.order = rng.permutation(self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        index = self.order[index]

        base_index = index // ((2 * self.pad + 1) * (2 * self.pad + 1) * 2)
        img, target = self.base[base_index]

        transform_index = index % ((2 * self.pad + 1) * (2 * self.pad + 1) * 2)
        flip_index = transform_index // ((2 * self.pad + 1) * (2 * self.pad + 1))
        crop_index = transform_index % ((2 * self.pad + 1) * (2 * self.pad + 1))
        crop_x = crop_index // (2 * self.pad + 1)
        crop_y = crop_index % (2 * self.pad + 1)

        if flip_index:
            img = torchvision.transforms.functional.hflip(img)
        img = torchvision.transforms.functional.pad(img, self.pad)
        img = torchvision.transforms.functional.crop(img, crop_x, crop_y, 32, 32)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class CIFAR100AUG(CIFAR10AUG):
    base_class = torchvision.datasets.CIFAR100
