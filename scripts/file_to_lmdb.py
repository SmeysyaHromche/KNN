# Script for converting a text file to LMDB format, where each line is stored with its line index as the key.
# Context: Need for load trn/tst files with random access to each row for dataset building.
import argparse
from pathlib import Path

import lmdb


def txt_to_lmdb(
    input_path: Path,
    lmdb_path: Path,
    map_size: int = 2 * 1024**3,
    commit_every: int = 10000,
) -> None:
    env = lmdb.open(str(lmdb_path), map_size=map_size)

    txn = env.begin(write=True)
    count = 0

    try:
        with input_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.rstrip("\n")
                if not line:
                    continue

                key = str(idx).encode("utf-8")
                value = line.encode("utf-8")

                txn.put(key, value)
                count += 1

                if count % commit_every == 0:
                    txn.commit()
                    txn = env.begin(write=True)

        txn.put(b"length", str(count).encode("utf-8"))
        txn.commit()

        print(f"Stored {count} rows in {lmdb_path}")

    except Exception:
        txn.abort()
        raise
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Store file lines in LMDB with row index as key.")
    parser.add_argument("input_file", type=Path, help="Input file")
    parser.add_argument("output_lmdb", type=Path, help="Output LMDB directory")
    parser.add_argument(
        "--map-size-gb",
        type=float,
        default=2.0,
        help="LMDB map size in GB",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=10000,
        help="Commit every N rows",
    )

    args = parser.parse_args()

    map_size = int(args.map_size_gb * 1024**3)

    txt_to_lmdb(
        input_path=args.input_file,
        lmdb_path=args.output_lmdb,
        map_size=map_size,
        commit_every=args.commit_every,
    )


if __name__ == "__main__":
    main()