import argparse
import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

logging.getLogger().setLevel(logging.INFO)


def read_image(path: str) -> np.ndarray:
    image = plt.imread(path)
    return image


def plot_image(image: np.ndarray, title: str = "Image to compress") -> None:
    plt.title(title)
    plt.imshow(image)
    plt.show()


def plot_image_channels(image: np.ndarray) -> None:
    logging.info(f"The shape of the image is {image.shape}")
    _, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(image, cmap="gray")
    ax[0, 0].set_title("Image to compress")
    ax[0, 1].imshow(image[:, :, 0], cmap="gray")
    ax[0, 1].set_title("R")
    ax[1, 0].imshow(image[:, :, 1], cmap="gray")
    ax[1, 0].set_title("G")
    ax[1, 1].imshow(image[:, :, 2], cmap="gray")
    ax[1, 1].set_title("B")
    plt.show()


def apply_svd(
    image: np.ndarray, channel: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, Sigma, Vt = np.linalg.svd(image[:, :, channel])
    # Sigma is already sorted in descending order as shown in the documentation
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    logging.info(f"The shape of Sigma is {Sigma.shape}")
    logging.info(f"The shape of U is {U.shape}")
    logging.info(f"The shape of Vt is {Vt.shape}")
    return U, Sigma, Vt


def compress_channel_svd(image: np.ndarray, channel: int, k: int) -> np.ndarray:
    U, Sigma, Vt = apply_svd(image, channel)
    compression = U[:, :k] @ np.diag(Sigma[:k]) @ Vt[:k, :]
    plt.title(f"Compressed channel={channel}, k ={k}")
    plt.imshow(compression, cmap="gray")
    plt.show()
    return compression


def compress_image_svd(image: np.ndarray, k: int) -> np.ndarray:
    compressed_image = np.zeros(image.shape)
    for channel in range(image.shape[-1]):
        compressed_image[:, :, channel] = compress_channel_svd(image, channel, k)
    logging.info(f"The shape of the compressed image is {compressed_image.shape}")
    return compressed_image


def plot_compressed_image(compressed_image: np.ndarray, k: int) -> None:
    # Set min pixel value to 0
    compressed_image = compressed_image - np.min(compressed_image)
    # Set max pixel value to 255
    compressed_image = compressed_image * 255 / np.max(compressed_image)
    # Set integer pixel values
    compressed_image = compressed_image.astype("int")
    # Notice that there are some color artifacts due the compression of
    # each color channel independently
    plt.title(f"k ={k}")
    plt.imshow(compressed_image)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse compression parameters")
    parser.add_argument(
        "-k",
        type=int,
        help="number of selected columns of the U matrix and selected rows of Vt",
        default=10,
    )
    parser.add_argument(
        "-p",
        "--image-path",
        type=str,
        help="path of the image to compress",
        required=True,
    )
    return parser.parse_args()


def compute_psnr(image: np.ndarray, compressed_image: np.ndarray, k: int) -> float:
    Rmax2 = 255**2
    e2 = np.linalg.norm(image - compressed_image) ** 2 / (605 * 605)
    psnr = 10 * np.log10(Rmax2 / e2)
    logging.info(f"For k = {k} the PSNR = {psnr}")
    return psnr


if __name__ == "__main__":
    args = parse_args()
    image = read_image(args.image_path)
    plot_image(image)
    plot_image_channels(image)
    compressed_image = compress_image_svd(image, args.k)
    plot_compressed_image(compressed_image, args.k)
    compute_psnr(image, compressed_image, args.k)
