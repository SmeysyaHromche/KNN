# TODO: delete test
import lmdb


def read_first_10(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    with env.begin() as txn:
        for i in range(10):
            key = str(i).encode("utf-8")
            value = txn.get(key)

            if value is None:
                print(f"{i}: <NOT FOUND>")
            else:
                print(f"{i}: {value}")
                print(f"{i}: {value.decode('utf-8')}")

    env.close()


def read_img_10(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    keys = ["241-r000-l020.jpg", "impact-107352-r13-l006.jpg", "read-000001-r1-l000.jpg"]
    with env.begin() as txn:
        for i, key_str in enumerate(keys):
            key = key_str.encode("utf-8")
            value = txn.get(key)

            if value is None:
                print(f"{i}: <NOT FOUND>")
            else:
                print(f"{i}: {value[:10]}")
    env.close()


if __name__ == "__main__":
    
    read_first_10("/home/xkukht01/Dev/KNN/.data/.meta")
    #read_img_10("/home/xkukht01/Dev/KNN/.data/.data")