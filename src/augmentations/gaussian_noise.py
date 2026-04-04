import numpy as np

from augmentations.base import Transform


class GaussianNoise(Transform):
    """
    Apply gaussian noise to an image.

    This transformation adds gaussian noise with a given mean
    and standard deviation to the input image.
    """
    def __init__(self, probability: float = 1.0, mean: float = 0.0, std: float = 1.0) -> None:
        """
        Initialize the gaussian noise transformation.

        :param probability: Probability that the transformation will be applied.
        :param mean: Mean value of the gaussian distribution.
        :param std: Standard deviation of the gaussian distribution.
        """
        super().__init__(probability)
        self.mean = mean
        self.std = std

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Add gaussian noise to the input image.

        :param image: Input image to add the gaussian noise to.
        """
        noise = np.random.normal(self.mean, self.std, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise

        return np.clip(noisy_image, 0, 255).astype(np.uint8)
