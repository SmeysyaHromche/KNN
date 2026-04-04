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
    # example usage
    read_key("data.lmdb", "impact-107352-r13-l006.jpg")