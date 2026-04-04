import numpy as np
from augmentations.base import Transform


class Compose:
    """
    Compose multiple transformations into a single pipeline.

    Each transformation is applied sequentially to the input image.
    The transformations are expected to be callable objects - instances
    of subclasses of ``Transform``

    :param transforms: Sequence of transformations to apply.
    """

    def __init__(self, transforms: list[Transform]) -> None:
        """
        Initialize the composition.

        :param transforms: Sequence of transformations.
        """
        self.transforms = transforms

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all transformations to the image.

        :param image: Input image to transform.
        :return: Transformed image after all operations.
        """
        for t in self.transforms:
            image = t(image)
        return image
