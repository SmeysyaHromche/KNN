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
                print(f"{i}: {value.decode('utf-8')}")

    env.close()


if __name__ == "__main__":
    read_first_10("data.lmdb")