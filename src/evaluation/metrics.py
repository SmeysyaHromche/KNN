import Levenshtein


class OCRMetrics:
    """
    Compute different metrics such as WER (Word Error Rate)
    and CER (Character Error Rate).

    Use either `update()` or `update_batch()` to update the
    inner state of the metrics. Once you want to retrieve the
    measured metrics, call the `compute()` method.
    """
    def __init__(self) -> None:
        self.char_errors = 0
        self.char_total = 0
        self.word_errors = 0
        self.word_total = 0

    def update_batch(self, predicted: list[str], target: list[str]) -> None:
        """
        Update the inner state of the instance in a batch (with multiple
        predicted/target strings).

        :param predicted: List of predictions
        :param target: List of labels (ground truth)
        """
        for p, t in zip(predicted, target):
            self.update(p, t)

    def update(self, predicted: str, target: str) -> None:
        """
        Update the inner state of the instance.

        :param predicted: String that the model predicted
        :param target: Label (ground truth)
        """
        # Character level distances
        self.char_errors += Levenshtein.distance(predicted, target)
        self.char_total += len(target)

        # Word level distances
        predicted_words = predicted.split()
        target_words = target.split()

        self.word_errors += Levenshtein.distance(predicted_words, target_words)
        self.word_total += len(target.split())

    def compute(self, use_percentages: bool = False) -> dict[str: float]:
        """
        Compute the CER and WER metrics. Their values are in the interval <0, 1>.
        If you want percentages, you can use the `use_percentages` argument.

        :param use_percentages: If 'True' the returned value will be in percentages.
        :returns: A dictionary with the computed metrics.
        """
        cer = self.char_errors / self.char_total if self.char_total > 0 else 0
        wer = self.word_errors / self.word_total if self.word_total > 0 else 0

        return {
            "cer": cer * 100 if use_percentages else cer,
            "wer": wer * 100 if use_percentages else wer,
        }

    def reset(self) -> None:
        """
        Reset the metrics.
        """
        self.__init__()
