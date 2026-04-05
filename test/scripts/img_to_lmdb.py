import io
import os
import shutil
import lmdb


def image_to_lmdb(image_path: str, image_name: str, lmdb_path: str) -> None:
    # Read jpg as raw bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Open LMDB and store image
    env = lmdb.open(lmdb_path, map_size=10**9)

    with env.begin(write=True) as txn:
        key = image_name.encode("utf-8")
        value = image_bytes
        txn.put(key, value)

    env.close()
    print(f"Stored image '{image_name}' in LMDB.")


if __name__ == "__main__":
    lmdb_path = "/home/xkukht01/Dev/KNN/.data/.data"
    # Check LMDB directory
    if os.path.exists(lmdb_path):
        answer = input(f"LMDB path '{lmdb_path}' already exists. Remove it? (y/n): ").strip().lower()

        if answer == "y":
            shutil.rmtree(lmdb_path)
            print("Old LMDB removed.")
        else:
            print("Operation cancelled.")
            exit()

    image_to_lmdb(
        image_path="/home/xkukht01/Dev/KNN/src/learn/database/test/241-r000-l020.jpg",
        image_name="241-r000-l020.jpg",
        lmdb_path=lmdb_path,
    )

    image_to_lmdb(
        image_path="/home/xkukht01/Dev/KNN/src/learn/database/test/impact-107352-r13-l006.jpg",
        image_name="impact-107352-r13-l006.jpg",
        lmdb_path=lmdb_path,
    )

    image_to_lmdb(
        image_path="/home/xkukht01/Dev/KNN/src/learn/database/test/read-000001-r1-l000.jpg",
        image_name="read-000001-r1-l000.jpg",
        lmdb_path=lmdb_path,
    )