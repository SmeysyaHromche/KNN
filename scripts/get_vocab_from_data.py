"""
Script to create the vocabulary (all possible characters) from the dataset's labels.

Expected label's structure:
<image-name> <text>
<image-name> <text>
...

CLI parameters:
    -i/--input_file <file> - can be used multiple times to read the labels and
                    create the vocabulary
    -o/--output_file <file> - where to write the vocabulary
    -e/--extend_vocab - when used, the current vocabulary in the provided output
                    file will be extended by the vocabulary found in input files
"""

import argparse
from pathlib import Path

# Special tokens needed by the decoder
SPECIAL_TOKENS = ["<bos>", "<eos>", "<pad>"]


def load_stored_vocabulary(file: Path) -> set[chr]:
    """
    Load the current vocabulary, so it can be extended.

    :param file: Filename of the current vocabulary
    :returns: The current vocabulary as a set
    """
    with open(file, "r", encoding="utf-8") as f:
        # The vocabulary is on character level, so only the SPECIAL_TOKENS can be longer
        # than 1 character long, so they are skipped here and written once again to the
        # beginning of the output file.
        cur_vocab = set(line for line in f.read().splitlines() if len(line) == 1)

    return cur_vocab

def extract_vocabulary(
    input_files: list[Path], output_file: Path, extend_vocab: bool = False
) -> None:
    """
    Extract the vocabulary from the input files. The expected structure is as follows:
    <image-name> <text>
    The '<image-name> ' is skipped and the '<text>' is parsed for the characters.

    :param input_files: List of all the input files to scrape the vocabulary from
    :param output_file: Name of the output file to store the vocabulary
    :param extend_vocab: Bool parameter to change whether to extend current vocabulary or not
    """
    vocabulary = set()

    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                # Skip the '<image-name> ' part
                line = line.partition(" ")[2].replace("\n", "")
                vocabulary.update(line)

    if extend_vocab:
        cur_vocab = load_stored_vocabulary(output_file)
        vocabulary = vocabulary | cur_vocab

    vocabulary = sorted(vocabulary)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(SPECIAL_TOKENS))
        f.write("\n")
        f.write("\n".join(vocabulary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        action="append",
        type=Path,
        required=True,
        help="Input files to extract the vocabulary from",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=Path,
        required=True,
        help="Output file, where the vocabulary is stored",
    )
    parser.add_argument(
        "-e",
        "--extend_vocab",
        action="store_true",
        help="Extend existing vocabulary in output file",
    )

    args = parser.parse_args()

    extract_vocabulary(args.input_file, args.output_file, args.extend_vocab)
