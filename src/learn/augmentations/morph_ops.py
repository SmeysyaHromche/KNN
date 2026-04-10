import random
import numpy as np
import cv2 as cv

from src.learn.augmentations.base import Transform


class RandomMorphology(Transform):
    """
    Apply a random morphological transformation to an image.

    This transformation randomly applies either **erosion** or **dilatation**
    to the input image using a square kernel of randomly chosen odd size.
    """

    def __init__(self, probability: float = 1.0, kernel_size_range : tuple[int, int] = (1, 3)) -> None:
        """
        Initialize the random morphological transformation.

        :param probability: Probability that the transformation will be applied when calling the transformation.
        :param kernel_size_range: Tuple indicating the min and max size of the square kernel.
        """
        super().__init__(probability)
        self.kernel_size_range = kernel_size_range

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply a random erosion or dilatation on the input image.

        Since these operations do the opposites, one operation will be chosen
        for the transformation. The kernel size is randomly selected from the
        range and is guaranteed to be an odd integer.

        :param image: Input image to apply the morphological transformation to.
        """
        k = random.randrange(self.kernel_size_range[0], self.kernel_size_range[1] + 1, 2)
        kernel = np.ones((k, k), np.uint8)

        option = random.choice(["erosion", "dilatation"])
        
        if option == "erosion":
            return cv.erode(image, kernel, iterations=1)
        else:
            return cv.dilate(image, kernel, iterations=1)
