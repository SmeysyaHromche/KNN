from .ocrdataset import OcrDataset
#!/usr/bin/env python3
import lmdb


def read_key(lmdb_path, key_str):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    with env.begin() as txn:
        key = key_str.encode("utf-8")
        value = txn.get(key)

        if value is None:
            print(f"Key '{key_str}' not found")
            return

        print("=== BYTES ===")
        print(value)
        print()

        print("=== STRING (utf-8) ===")
        try:
            print(value.decode("utf-8"))
        except UnicodeDecodeError:
            print("Cannot decode as UTF-8 (likely binary data)")

    env.close()


if __name__ == "__main__":
    read_key("/mnt/matylda1/ikiss/data/knn_ocr/impact/lines_48-1.15.lmdb", "")