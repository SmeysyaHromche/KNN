from pathlib import Path


class Tokenizer:
    """
    Class to encode and decode text using some vocabulary.

    :param vocabulary_file: Path to the file containing the vocabulary
    """
    def __init__(self, vocabulary_file: Path) -> None:
        with open(vocabulary_file, "r", encoding="utf-8") as f:
            self.tokens = [line.rstrip("\n") for line in f]

        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for i, t in enumerate(self.tokens)}

    def encode(self, text: str) -> list[int]:
        """
        Encode some text into a list of indices of each character.
        Each index represents the index of the line in the vocabulary
        file where the used character is located.

        :param text: Text to encode into the list of indices
        :returns: Encoded list
        """
        return [self.token_to_id[t] for t in text]

    def decode(self, ids: list[int]) -> str:
        """
        Decode the list of indexes back to a textual form.

        :param ids: List of the indices of each character
        :returns: Decoded string
        """
        tokens = [self.id_to_token[t] for t in ids]
        return "".join(tokens)
