import h5py
import numpy as np
import os
from PIL import Image
from PIL.Image import Image as Image_t
import matplotlib.pyplot as plt
import torch

from abc import ABC, abstractmethod
from typing import Type, Union


class storage_class(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.pos = 0
        self._locked = False
        
    @property
    @abstractmethod
    def dataset_size(self) -> int:
        """
        Return the number of (image, mask) pairs in the dataset
        """
        pass
    
    @abstractmethod
    def append(self, input_chunk: np.ndarray, target_chunk: np.ndarray) -> None:
        """
        Add a block of images `input_chunk` and masks `target_chunk` to the dataset
        """
        pass
    
    @abstractmethod
    def lock(self) -> None:
        """
        Close writing mode, open in reading mode
        """
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[Image_t, Image_t]:
        """
        Return the (image, mask) pair at index `idx`
        """
        pass

    
class storage_hdf5(storage_class):
    def __init__(self, config):
        super().__init__(config)
        
        self.dataset = h5py.File(f'{config["annotation_file"].removesuffix(".txt")}.h5', 'a')          
        self.dataset.create_dataset("input", shape=(config["dataset_size"], *config["target_shape"], 3), dtype=np.uint8)
        self.dataset.create_dataset("target",  shape=(config["dataset_size"], *config["target_shape"]), dtype=np.uint8)
        
    @property
    def dataset_size(self):
        assert self.dataset["input"].shape[:-1] == self.dataset["target"].shape
        assert self.dataset["target"].shape[0] == self.config["dataset_size"]
        return self.config["dataset_size"]
    
    def append(self, input_chunk, target_chunk):
        l_idx = self.pos * self.config["chunk_size"]
        r_idx = min(self.config["dataset_size"], l_idx + self.config["chunk_size"])
        self.dataset["input"][l_idx:r_idx] = input_chunk[:r_idx-l_idx]
        self.dataset["target"][l_idx:r_idx] = target_chunk[:r_idx-l_idx]
        self.pos += 1
    
    def lock(self):
        self.dataset.close()
        self.dataset = h5py.File(f'{self.config["annotation_file"].removesuffix(".txt")}.h5', 'r')
        self._locked = True
    
    def __getitem__(self, idx):
        if not self._locked:
            raise AttributeError("HDF5 file is not locked. Access denied.")
        return Image.fromarray(self.dataset["input"][idx]), Image.fromarray(self.dataset["target"][idx])

    
class storage_memmap(storage_class):
    def __init__(self, config):
        super().__init__(config)
        
        path = config["annotation_file"].removesuffix(".txt")
        os.makedirs(path, exist_ok=True)
        self.input_path = os.path.join(path, "input.npy")
        self.target_path = os.path.join(path, "target.npy")

        self.input = np.memmap(self.input_path, mode='w+', dtype=np.uint8, shape=(config["dataset_size"], *config["target_shape"], 3))
        self.target = np.memmap(self.target_path, mode='w+', dtype=np.uint8, shape=(config["dataset_size"], *config["target_shape"]))
    
    @property
    def dataset_size(self):
        assert self.input.shape[:-1] == self.target.shape
        assert self.input.shape[0] == self.config["dataset_size"]
        return self.config["dataset_size"]
                                 
    def append(self, input_chunk, target_chunk):
        l_idx = self.pos * self.config["chunk_size"]
        r_idx = min(self.config["dataset_size"], l_idx + self.config["chunk_size"])
        self.input[l_idx:r_idx] = input_chunk[:r_idx-l_idx]
        self.target[l_idx:r_idx] = target_chunk[:r_idx-l_idx]
        self.pos += 1
    
    def lock(self):
        del self.input
        del self.target
        self.input = np.memmap(self.input_path, mode='r', dtype=np.uint8, shape=(self.config["dataset_size"], *self.config["target_shape"], 3))
        self.target = np.memmap(self.target_path, mode='r', dtype=np.uint8, shape=(self.config["dataset_size"], *self.config["target_shape"]))
        self._locked = True
    
    def __getitem__(self, idx):
        if not self._locked:
            raise AttributeError("Memory-mapped file is not locked. Access denied.")
        return Image.fromarray(self.input[idx]), Image.fromarray(self.target[idx])

    
class storage_raw(storage_class):
    def __init__(self, config):
        super().__init__(config)
                                 
        path = config["annotation_file"].removesuffix(".txt")
        self.input_path = os.path.join(path, "Input")
        self.target_path = os.path.join(path, "Target")
        self.idx2input = lambda idx: os.path.join(self.input_path, f"input{idx}.jpg")
        self.idx2target = lambda idx: os.path.join(self.target_path, f"target{idx}.png")
        
        os.makedirs(path, exist_ok=True)
        os.makedirs(self.input_path, exist_ok=False)
        os.makedirs(self.target_path, exist_ok=False)
    
    @property
    def dataset_size(self):
        assert len(os.listdir(self.input_path)) == len(os.listdir(self.target_path))
        assert len(os.listdir(self.input_path)) == self.config["dataset_size"]
        return self.config["dataset_size"]
                                 
    def append(self, input_chunk, target_chunk):
        l_idx = self.pos * self.config["chunk_size"]
        r_idx = min(self.config["dataset_size"], l_idx + self.config["chunk_size"])
                                 
        for idx in range(l_idx, r_idx):
            Image.fromarray(input_chunk[idx-l_idx], "RGB").save(self.idx2input(idx))
            Image.fromarray(target_chunk[idx-l_idx], 'L').save(self.idx2target(idx))

        self.pos += 1
                                 
    def lock(self):
        self._locked = True
                                 
    def __getitem__(self, idx):
        if not self._locked:
            raise AttributeError("Raw file is not locked. Access denied.")
        img = Image.open(self.idx2input(idx)).convert("RGB")
        target = Image.open(self.idx2target(idx)).convert('L')
        return img, target


def renumerate_target(target: np.ndarray, label: int) -> np.ndarray:
    """
    Renumber the segmentation masks.
    In the original dataset, the following labels are used {1: object, 2: background, 3: ignore_index},
    where the object is defined by the variable `label` from the set {1 (cat), 2 (dog)}.
    We convert the values to a unified format {0: background, 1: cat, 2: dog, 255: ignore_index}.
    `ignore_index` corresponds to mask pixels that are not included in the metric calculations
    (these are object contours that are hard to predict accurately).
    """
    target_map = {1: label, 2: 0, 3: 255}
    indexer = np.array(list(target_map.values()))
    target = indexer[(target - target.min())]
    return target


def colorize(data):
    color_map = np.array([
        [0, 0, 0],        # 0: black
        [0, 0, 255],      # 1: blue
        [255, 0, 0],      # 2: red
    ])

    result = np.zeros((*data.shape, 3), dtype=np.uint8)
    for label, color in enumerate(color_map):
        result[data == label] = color
    result[data == 255] = [255, 255, 255]  # white
    return result


# Function for drawing images
def draw(pair: tuple[torch.Tensor, torch.Tensor], t_dict: dict, prediction: torch.Tensor = None, log: bool = False):
    """
    `pair` contains the (image, segmentation mask) pair.
    `prediction` contains the model-predicted segmentation mask.
    `log` flag indicating whether to log to tensorboard.
    """
    img, target = pair
    num_images = 3
    if prediction is None:
        num_images = 2
    
    f, ax = plt.subplots(1, num_images, figsize=(8*num_images, 8))
    ax[0].imshow(t_dict["backward_input"](img))
    ax[0].axis('off')
    ax[1].imshow(colorize(t_dict["backward_target"](target)))
    ax[1].axis('off')

    if prediction is not None:
        prediction[target == 255] = 255
        ax[2].imshow(colorize(t_dict["backward_target"](prediction)))
        ax[2].axis('off')
        
    plt.tight_layout()
    if log:
        return f
    plt.show()