import cv2 as cv
import random
import numpy as np

from src.learn.augmentations.base import Transform

class GaussianBlur(Transform):
    """
    Apply Gaussian blur to an image with a given probability.

    Gaussian blur helps to simulate slight camera or scanning imperfections,
    which can make OCR models more robust to real-world data.
    """

    def __init__(self, probability: float = 1.0, kernel_size_range: tuple[int, int] = (5, 20)) -> None:
        """
        Initialize the Gaussian blur transformation.

        :param probability: Probability of applying the blur.
        :param kernel_size_range: Minimum and maximum kernel size for the Gaussian blur
        (both must be odd integers).
        """

        super().__init__(probability)
        self.kernel_size_range = kernel_size_range

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to the input image.

        A random odd kernel size is chosen between `min_kernel_size` and
        `max_kernel_size` to vary the blur intensity each time the transformation
        is applied.

        :param image: Input image (NumPy array) to be blurred.
        :return: Blurred image as a NumPy array.
        """

        min_k, max_k = self.kernel_size_range

        # Select a random odd kernel size (1, 3, ..., max_kernel_size)
        k_size = random.choice(range(min_k, max_k + 1, 2))

        # Apply Gaussian blur with the chosen kernel size
        return cv.GaussianBlur(image, (k_size, k_size), 0)
