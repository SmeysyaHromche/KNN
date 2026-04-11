import argparse
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly split file into train and validation sets (streaming)."
    )
    parser.add_argument("-t", "--train", type=float, required=True)
    parser.add_argument("-v", "--valid", type=float, required=True)
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, valid_ratio: float) -> None:
    if train_ratio < 0 or valid_ratio < 0:
        raise ValueError("Values for -t and -v must be non-negative")

    if abs((train_ratio + valid_ratio) - 1.0) > 1e-9:
        raise ValueError("The sum of -t and -v must be exactly 1")


def split_file_random(
    input_path: Path,
    output_name: str,
    train_ratio: float,
    seed: int | None = None,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if not input_path.is_file():
        raise ValueError(f"Provided path is not a file: {input_path}")

    if seed is not None:
        random.seed(seed)

    train_path = input_path.parent / f"{output_name}.trn"
    valid_path = input_path.parent / f"{output_name}.vld"

    train_count = 0
    valid_count = 0

    with (
        input_path.open("r", encoding="utf-8") as src,
        train_path.open("w", encoding="utf-8") as train_file,
        valid_path.open("w", encoding="utf-8") as valid_file,
    ):
        for line in src:
            if random.random() < train_ratio:
                train_file.write(line)
                train_count += 1
            else:
                valid_file.write(line)
                valid_count += 1

    total = train_count + valid_count

    print(f"Total lines: {total}")
    print(f"Train lines: {train_count}")
    print(f"Valid lines: {valid_count}")


def main():
    args = parse_args()
    validate_ratios(args.train, args.valid)
    split_file_random(Path(args.path), args.name, args.train, args.seed)


if __name__ == "__main__":
    main()