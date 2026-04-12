from pathlib import Path


class Tokenizer:
    """
    Class to encode and decode text using some vocabulary.

    :param vocabulary_file: Path to the file containing the vocabulary
    """
    def __init__(self, vocabulary_file: Path) -> None:
        with open(vocabulary_file, "r", encoding="utf-8") as f:
            self.tokens = [line.rstrip("\n") for line in f]

        self.vocab_size = sum(1 for _ in self.tokens)

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
    
    def encode_special_token(self, token: str) -> int:
        """
        Encode special token needed by the transformer (e.g. '<bos>') into an id.
        If the special token was encoded with the `encode()` function, it
        would return the ids for each character ('<' -> id1, 'b' -> id2, ...).

        :param token: Special token to encode
        :returns: Encoded special token
        """
        return self.token_to_id[token]

    def decode(self, ids: list[int], remove_bos: bool = True, remove_after_eos: bool = True) -> str:
        """
        Decode the list of indexes back to a textual form.

        :param ids: List of the indices of each character
        :param remove_after_eos: When set to 'True' the decoding will remove everything
            that comes after '<eos>' (End Of Sequence) including the '<eos>' token.
        :returns: Decoded string
        """
        tokens = []
        eos = self.encode_special_token("<eos>")
        bos = self.encode_special_token("<bos>")

        for t in ids:
            if remove_bos and t == bos:
                continue
            if remove_after_eos and t == eos:
                break
            tokens.append(self.id_to_token[t])

        return "".join(tokens)
    
    def get_vocab_size(self) -> int:
        """
        Getter method for the size of the used vocabulary (how many characters + special
        transformer tokens are in the vocabulary).

        :returns: The size of the vocabulary
        """
        return self.vocab_size
