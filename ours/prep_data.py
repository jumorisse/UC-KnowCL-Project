from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import os
import random


def augment_image(image, augmentation_probabilities):
    augmented_image = image.copy()

    # Horizontal flip
    if random.random() < augmentation_probabilities["horizontal_flip"]:
        augmented_image = ImageOps.mirror(augmented_image)

    # Vertical flip
    if random.random() < augmentation_probabilities["vertical_flip"]:
        augmented_image = ImageOps.flip(augmented_image)

    # Brightness
    if random.random() < augmentation_probabilities["brightness"]:
        # changes brightness by a factor between 0.5 and 1.0
        # (0.5 makes the image darker, 1.0 leaves the image unchanged)
        # https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.Brightness
        enhancer = ImageEnhance.Brightness(augmented_image)
        factor = 0.5 + random.random() * 0.5
        augmented_image = enhancer.enhance(factor)

    # Gaussian noise (generates noises for all channels of the image)
    if random.random() < augmentation_probabilities["gaussian_noise"]:
        # generate noise for all channels
        noise = np.random.normal(
            0, 1, (augmented_image.size[1], augmented_image.size[0], 3)
        )
        # add noise to image
        augmented_image = Image.fromarray(
            np.array(augmented_image) + noise.astype(np.uint8)
        )

    # Gaussian blur
    if random.random() < augmentation_probabilities["gaussian_blur"]:
        augmented_image = augmented_image.filter(ImageFilter.GaussianBlur(1))

    return augmented_image


def produce_augmented(augmented_data_path, original_data_path):
    augmentation_probabilities = {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.5,
        "brightness": 0.2,
        "gaussian_noise": 0.1,
        "gaussian_blur": 0.1,
    }

    if not os.path.exists(augmented_data_path):
        os.makedirs(augmented_data_path)

    images = os.listdir(original_data_path)
    images = [image for image in images if image.endswith(".png")]

    for image in images:
        image_name = image
        image_path = os.path.join(original_data_path, image)
        image = Image.open(image_path)

        augmented_image = augment_image(image, augmentation_probabilities)

        augmented_image_path = os.path.join(augmented_data_path, image_name)
        augmented_image.save(augmented_image_path)


def produce_combined(combined_data_path, original_data_path, cloudy_data_path):
    augmentation_probabilities = {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.5,
        "brightness": 0.2,
        "gaussian_noise": 0.1,
        "gaussian_blur": 0.1,
    }
    if not os.path.exists(combined_data_path):
        os.makedirs(combined_data_path)

    images = os.listdir(original_data_path)
    images = [image for image in images if image.endswith(".png")]

    for image in images:
        image_name = image
        original_image_path = os.path.join(original_data_path, image)
        original_image = Image.open(original_image_path)

        cloudy_image_path = os.path.join(cloudy_data_path, image)
        cloudy_image = Image.open(cloudy_image_path)

        if random.random() < 0.5:
            image = original_image
        else:
            image = cloudy_image

        augmented_image = augment_image(image, augmentation_probabilities)

        augmented_image_path = os.path.join(combined_data_path, image_name)
        augmented_image.save(augmented_image_path)


if __name__ == "__main__":
    original_data_path = "../data/new_york/satellite_image/zl15_224/"
    cloudy_data_path = "../data/new_york_cloudy/satellite_image/zl15_224/"
    augmented_data_path = "../data/new_york_augmented/satellite_image/zl15_224/"
    combined_data_path = "../data/new_york_combined/satellite_image/zl15_224/"

    produce_augmented(augmented_data_path, original_data_path)
    produce_combined(combined_data_path, original_data_path, cloudy_data_path)
