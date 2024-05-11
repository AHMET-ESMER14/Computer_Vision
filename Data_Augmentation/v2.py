import cv2
import albumentations as A
from matplotlib import pyplot as plt

image = cv2.imread("C://Users//Monster//Desktop//1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def apply_rotation(image, angle):
    transform = A.Rotate(limit=angle, p=1)
    transformed = transform(image=image)
    return transformed["image"]

def apply_flip(image):
    transform = A.VerticalFlip(p=0.5)
    transformed = transform(image=image)
    return transformed["image"]

def apply_crop(image, crop_height, crop_width):
    transform = A.RandomCrop(height=crop_height, width=crop_width, p=1)
    transformed = transform(image=image)
    return transformed["image"]

def apply_color_change(image):
    transform = A.RandomBrightnessContrast(p=0.5)
    transformed = transform(image=image)
    return transformed["image"]

def apply_zoom_shift(image):
    transform = A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, p=1)
    transformed = transform(image=image)
    return transformed["image"]

def apply_noise(image):
    transform = A.GaussNoise(p=0.5)
    transformed = transform(image=image)
    return transformed["image"]

def apply_resize(image, new_height, new_width):
    transform = A.Resize(height=new_height, width=new_width)
    transformed = transform(image=image)
    return transformed["image"]

def apply_elastic_deformation(image):
    transform = A.ElasticTransform(p=1)
    transformed = transform(image=image)
    return transformed["image"]

def apply_geometric_transform(image):
    transform = A.IAAPerspective(p=1)
    transformed = transform(image=image)
    return transformed["image"]

def apply_perspective(image):
    transform = A.Perspective(p=1)
    transformed = transform(image=image)
    return transformed["image"]

def apply_blur(image):
    transform = A.Blur(blur_limit=3, p=1)
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

rotation_image = apply_rotation(image, angle=45)
visualize(image, rotation_image, "Image Rotation")

flip_image = apply_flip(image)
visualize(image, flip_image, "Reflection and Transformation")

crop_image = apply_crop(image, crop_height=200, crop_width=200)
visualize(image, crop_image, "Cropping")

color_change_image = apply_color_change(image)
visualize(image, color_change_image, "Color Change")

zoom_shift_image = apply_zoom_shift(image)
visualize(image, zoom_shift_image, "Zoom and Shift")

noise_image = apply_noise(image)
visualize(image, noise_image, "Adding Noise")

resize_image = apply_resize(image, new_height=300, new_width=300)
visualize(image, resize_image, "Resizing Image")

elastic_deformation_image = apply_elastic_deformation(image)
visualize(image, elastic_deformation_image, "Elastic Deformations")

# geometric_transform_image = apply_geometric_transform(image)
# visualize(image, geometric_transform_image, "Geometric Transformations")

perspective_image = apply_perspective(image)
visualize(image, perspective_image, "Perspective Transformation")

blur_image = apply_blur(image)
visualize(image, blur_image, "Adding Random Blur")
