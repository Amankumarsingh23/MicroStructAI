"""Session 1: represent a tiny grayscale image as a NumPy array."""

import numpy as np


def main():
    image = np.array(
        [[240, 230, 45], [220, 125, 60], [235, 205, 30]], dtype=np.uint8
    )
    print("Image shape:", image.shape)
    print("Data type:", image.dtype)
    print("Pixel at row 0, column 1:", image[0, 1])
    print("Mean intensity:", image.mean())


if __name__ == "__main__":
    main()
