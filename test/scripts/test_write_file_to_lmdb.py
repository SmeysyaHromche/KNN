import lmdb
import os
import shutil

def txt_to_lmdb(txt_path, lmdb_path):
    if os.path.exists(lmdb_path):
        answer = input(f"LMDB path '{lmdb_path}' already exists. Remove it? (y/n): ").strip().lower()
        
        if answer == "y":
            shutil.rmtree(lmdb_path)
            print("Old LMDB removed.")
        else:
            print("Operation cancelled.")
            return

    env = lmdb.open(lmdb_path, map_size=10**9)
    with env.begin(write=True) as txn:
        with open(txt_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.rstrip("\n")
                key = str(idx).encode("utf-8")
                value = line.encode("utf-8")
                txn.put(key, value)

    env.close()


if __name__ == "__main__":
    txt_to_lmdb("/home/xkukht01/Dev/KNN/.log/test2.trn", "/home/xkukht01/Dev/KNN/.data/.meta")