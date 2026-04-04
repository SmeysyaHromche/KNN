import lmdb
from torch.utils.data import Dataset


class OcrDataset(Dataset):
    """
    PyTorch Dataset for OCR data stored in LMDB format.
    Args:
        path_to_db (str): Path to the LMDB database containing image bytes.
        path_to_meta_db (str): Path to the LMDB database containing metadata (key-label pairs).
    """
    def __init__(self, path_to_db: str, path_to_meta_db: str):
        self.path_to_db = path_to_db
        self.path_to_meta_db = path_to_meta_db

        self._data_db = None
        self._meta_db = None
        self._length = None

        self._init_length()

    """
    Open an LMDB database with optimal settings for read-only access.
    Args:
        path (str): Path to the LMDB database.
    Returns:
        lmdb.Environment: The opened LMDB environment.
    """
    def _open_db(self, path: str):
        return lmdb.open(
            path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

    """
    Get the data LMDB environment, opening it if it hasn't been opened yet.
    Returns:
        lmdb.Environment: The data LMDB environment.
    """
    def _get_data_db(self):
        if self._data_db is None:
            self._data_db = self._open_db(self.path_to_db)
        return self._data_db
    """
    Get the metadata LMDB environment, opening it if it hasn't been opened yet
    Returns:
        lmdb.Environment: The metadata LMDB environment.
    """
    def _get_meta_db(self):
        if self._meta_db is None:
            self._meta_db = self._open_db(self.path_to_meta_db)
        return self._meta_db

    """
    Get length of the dataset.
    """
    def _init_length(self):
        env = self._open_db(self.path_to_meta_db)
        try:
            with env.begin() as txn:
                self._length = txn.stat()["entries"]
        finally:
            env.close()

    """
    Close any open LMDB environments to free resources.
    """
    def close_resources(self):
        if self._data_db is not None:
            self._data_db.close()
            self._data_db = None

        if self._meta_db is not None:
            self._meta_db.close()
            self._meta_db = None

    
    """
    Retrieve a value-label pair from the dataset based on the index.
    Args:
        idx (int): The index of the sample to retrieve.
    Returns:
        tuple: A tuple containing the image bytes and the corresponding label.
    """
    def get_from_db(self, db, key: bytes):
        with db.begin() as txn:
            return txn.get(key)

    """
    Convert bytes data from the metadata LMDB into a key-label pair.
    Args:        
        bytes_data (bytes): The raw bytes data retrieved from the metadata LMDB.
    Returns:
        tuple: A tuple containing the key (str) and label (str).
    """
    def bytes_to_key_label_pair(self, bytes_data):
        if bytes_data is None:
            return None, None

        string_data = bytes_data.decode("utf-8")
        key, label = string_data.split("\t", maxsplit=1)
        return key, label

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        meta_db = self._get_meta_db()
        data_db = self._get_data_db()

        meta_bytes = self.get_from_db(meta_db, str(idx).encode("utf-8"))
        key, label = self.bytes_to_key_label_pair(meta_bytes)

        if key is None:
            raise IndexError(f"Metadata not found for idx={idx}")

        value = self.get_from_db(data_db, key.encode("utf-8"))
        if value is None:
            raise KeyError(f"Image bytes not found for key={key!r}")

        return value, label