import lmdb
from torch.utils.data import Dataset, get_worker_info

class OcrDataset(Dataset):

    def __init__(self, path_to_db: str, path_to_meta_db: str):
        self.path_to_db = path_to_db
        self.path_to_meta_db = path_to_meta_db
        
        self._data_db = lmdb.open(self.path_to_db, readonly=True)
        self._meta_db = lmdb.open(self.path_to_meta_db, readonly=True)


    def close_resources(self):
        self._data_db.close()
        self._meta_db.close()

    def __len__(self):
        with self._data_db.begin() as txn:
            return txn.stat()["entries"]

    def get_from_db(self, db, key:bytes):
        with db.begin() as txn:
            return txn.get(key)
    
    def bytes_to_key_label_pair(self, bytes_data):
        if bytes_data is None:
            return None, None
        string_data = bytes_data.decode("utf-8")
        key, label = string_data.split("\t")
        return key, label
        
    
    def __getitem__(self, idx):
        meta_bytes = self.get_from_db(self._meta_db, str(idx).encode())
        key, label = self.bytes_to_key_label_pair(meta_bytes)
        value = self.get_from_db(self._data_db, key.encode())
        return value, label
