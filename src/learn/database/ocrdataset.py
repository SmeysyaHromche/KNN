import io
import lmdb
import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class OcrDataset(Dataset):
    """
    PyTorch Dataset for OCR data stored in LMDB format.
    
    Behavior:
        - Each sample consists of an image and its corresponding label.
        - Images are stored as bytes in one LMDB database, and metadata (key-label pairs) are stored in another LMDB database.
        - The dataset lazily loads images and labels.
    
    Args:
        path_to_db (str): Path to the LMDB database containing image bytes.
        path_to_meta_db (str): Path to the LMDB database containing metadata (key-label pairs).
        transform (callable, optional): Optional transform augmentations to apply to the images.
    """
    def __init__(self, path_to_db: str, path_to_meta_db: str, transform=None):
        self.path_to_db = path_to_db
        self.path_to_meta_db = path_to_meta_db
        # If no transformations are passed, convert it just to Tensor
        self.transform = transform if transform is not None else transforms.ToTensor()

        self._data_db = None
        self._meta_db = None
        self._length = None

        self._init_length()

    def _open_db(self, path: str) -> lmdb.Environment:
        """
        Open an LMDB database with optimal settings for read-only access.
        Args:
            path (str): Path to the LMDB database.
        Returns:
            lmdb.Environment: The opened LMDB environment.
        """
        return lmdb.open(
            path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=512,
            subdir=True
        )

    def _get_data_db(self) -> lmdb.Environment:
        """
        Get the data LMDB environment, opening it if it hasn't been opened yet.
        Returns:
            lmdb.Environment: The data LMDB environment.
        """
        if self._data_db is None:
            self._data_db = self._open_db(self.path_to_db)
        return self._data_db

    def _get_meta_db(self) -> lmdb.Environment:
        """
        Get the metadata LMDB environment, opening it if it hasn't been opened yet
        Returns:
            lmdb.Environment: The metadata LMDB environment.
        """
        if self._meta_db is None:
            self._meta_db = self._open_db(self.path_to_meta_db)
        return self._meta_db


    def _init_length(self) -> None:
        """
        Get length of the dataset.
        """
        env = self._open_db(self.path_to_meta_db)
        try:
            with env.begin() as txn:
                self._length = txn.stat()["entries"] - 1
        finally:
            env.close()
    
    def close_resources(self) -> None:
        """
        Close any open LMDB environments to free resources.
        """
        if self._data_db is not None:
            self._data_db.close()
            self._data_db = None
        if self._meta_db is not None:
            self._meta_db.close()
            self._meta_db = None

    def _get_from_db(self, db, key: bytes) -> bytes | None:
        """
        Retrieve a value-label pair from the dataset based on the index.
        Args:
            idx (int): The index of the sample to retrieve.
        Returns:
            tuple: A tuple containing the image bytes and the corresponding label.
        """
        with db.begin() as txn:
            return txn.get(key)

    def _bytes_to_key_label_pair(self, bytes_data: bytes) -> tuple[str, str]:
        """
        Convert bytes data from the metadata LMDB into a key-label pair.
        Args:        
            bytes_data (bytes): The raw bytes data retrieved from the metadata LMDB.
        Returns:
            tuple: A tuple containing the key (str) and label (str).
        """
        if bytes_data is None:
            raise ValueError("Bytes data cannot be None")

        string_data = bytes_data.decode("utf-8")
        parts = string_data.split(" ", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed metadata record: {string_data!r}")
        return parts[0], parts[1]

    def _get_img_bytes_and_label(self, idx: int) -> tuple[bytes, str]:
        """
        Get the image bytes and label for a given index.
        Args:
            idx (int): The index of the sample to retrieve.
        Returns:
            tuple: A tuple containing the image bytes and the corresponding label.
        """
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index out of bounds: {idx}")

        meta_db = self._get_meta_db()
        data_db = self._get_data_db()

        meta_bytes = self._get_from_db(meta_db, str(idx).encode("utf-8"))
        if meta_bytes is None:
            raise KeyError(f"Metadata not found for idx={idx}")

        key, label = self._bytes_to_key_label_pair(meta_bytes)
        if not key:
            raise ValueError(f"Empty key for idx={idx}")

        img_bytes = self._get_from_db(data_db, key.encode("utf-8"))
        if img_bytes is None:
            raise KeyError(f"Image bytes not found for key={key!r}")

        return img_bytes, label
    
    def _bytes_to_numpy_image(self, img_bytes: bytes) -> np.ndarray:
        """
        Convert image bytes to a NumPy array (H, W, C) in RGB order.
        Args:
            img_bytes (bytes): The raw image bytes.
        Returns:
            np.ndarray: The converted NumPy array.
        """
        if img_bytes is None:
            raise ValueError("Image bytes cannot be None")

        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(image)

    def get_img_tensor_from_img_bytes(self, img_bytes: bytes) -> Tensor:
        """
        Convert raw image bytes into a tensor using PIL and torchvision transforms.
        Args:
            img_bytes (bytes): The raw image bytes.
        Returns:
            torch.Tensor: The converted image tensor.
        """
        if img_bytes is None:
            raise ValueError("Image bytes cannot be None")

        image_np = self._bytes_to_numpy_image(img_bytes)
        
        # Apply augmentations (OpenCV or NumPy based)
        if self.transform:
            image_np = self.transform(image_np)

        # Ensure tensor conversion
        if not isinstance(image_np, Tensor):
            image_np = transforms.ToTensor()(image_np)

        return image_np

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        Returns:
            int: The number of samples in the dataset.
        """
        return self._length

    def __getitem__(self, idx: int) -> tuple[Tensor, str]:
        """
        Get a sample from the dataset at the specified index.
        Args:
            idx (int): The index of the sample to retrieve.
        Returns:
            tuple: A tuple containing the image tensor and the corresponding label.
        """
        img_bytes, label = self._get_img_bytes_and_label(idx)
        img_tensor = self.get_img_tensor_from_img_bytes(img_bytes)
        return img_tensor, label

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_data_db"] = None
        state["_meta_db"] = None
        return state
