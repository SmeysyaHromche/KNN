import lmdb


def read_key(lmdb_path, key_str):
    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )

    with env.begin() as txn:
        key = key_str.encode("utf-8")
        value = txn.get(key)

        if value is None:
            print(f"Key {key_str!r} not found")
            return

        print("=== BYTES ===")
        print(value[:200])
        
        open(key_str, "wb").write(value)

    env.close()

if __name__ == "__main__":
    read_key("/mnt/matylda1/ikiss/data/knn_ocr/impact/lines_48-1.15.lmdb", "impact-107352-r13-l006.jpg")
    read_key("/mnt/matylda1/ikiss/data/knn_ocr/read/lines_48-1.15.lmdb", "read-000001-r1-l000.jpg")
    read_key("/mnt/matylda1/ikiss/data/knn_ocr/rodrigo/lines_48-1.15.lmdb", "241-r000-l020.jpg")