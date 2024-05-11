import cv2
import albumentations as A
from matplotlib import pyplot as plt

image = cv2.imread("C://Users//Monster//Desktop//1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def apply_brightness_contrast(image):
    transform = A.RandomBrightnessContrast(p=0.5)
    transformed = transform(image=image)
    return transformed["image"]

def apply_horizontal_flip(image):
    transform = A.HorizontalFlip(p=0.5)
    transformed = transform(image=image)
    return transformed["image"]

def apply_shift_scale_rotate(image):
    transform = A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5)
    transformed = transform(image=image)
    return transformed["image"]

def visualize(image, transformed_image, title):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image)
    plt.title(title)
    plt.show()

brightness_contrast_image = apply_brightness_contrast(image)
visualize(image, brightness_contrast_image, "Changing Brightness and Contrast")

horizontal_flip_image = apply_horizontal_flip(image)
visualize(image, horizontal_flip_image, "Horizontal Flip")

shift_scale_rotate_image = apply_shift_scale_rotate(image)
visualize(image, shift_scale_rotate_image, "Shifting, Scaling, and Rotating")
