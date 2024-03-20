from PIL import Image
from patchify import patchify, unpatchify
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import segmentation_models as sm
from utils.my_utils import show_mask_with_label

# Define class labels
labels = ['unclassified', 'vegetation', 'no vegetation', 'water', 'cloud']

# Validation Dataset
val_img_dir = "C:/Users/keena/Downloads/CubeNet_dataset/validation/images/val/"
val_mask_dir = "C:/Users/keena/Downloads/CubeNet_dataset/validation/masks/val/"

# Define the priorities for each class
class_priorities = {
    'unclassified': 0,
    'vegetation': 1,
    'no vegetation': 1,
    'water': 1,
    'cloud': 0
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a saved model for testing.')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the saved model')
    args = parser.parse_args()

    # Load the saved model
    model = tf.keras.models.load_model(args.model_dir, compile=False)
    model.compile()

    # Print model summary
    print(model.summary())

    # Extract backbone name from model directory path
    model_dir_split = args.model_dir.split('/')
    model_name = model_dir_split[1]
    backbone_name = model_name.split('_')[1]
    preprocess_input = sm.get_preprocessing(backbone_name)


    # Get the list of images and masks
    img_list = os.listdir(val_img_dir)
    msk_list = os.listdir(val_mask_dir)

    img_list.sort()
    msk_list.sort()

    # Initialize lists and arrays
    image_numbers = []
    overall_priorities = []
    overall_priorities_original = []
    crop_images = []
    crop_masks = []
    reconstructed_images = []
    test_pred_argmax_array = []

    # Loop through all images
    for img_num in range(len(img_list)):
        full_img = cv2.imread(os.path.join(val_img_dir, img_list[img_num]), 1)
        full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)

        full_mask = cv2.imread(os.path.join(val_mask_dir, msk_list[img_num]), cv2.IMREAD_GRAYSCALE)

        # Crop the images and masks
        x, y = 0, 0
        h, w = full_img.shape[0] // 64 * 64, full_img.shape[1] // 64 * 64
        crop_img = full_img[y:y+h, x:x+w]
        crop_mask = full_mask[y:y+h, x:x+w]

        # Patchify the cropped images and masks
        img_patches = patchify(crop_img, (64, 64, 3), step=64)
        mask_patches = patchify(crop_mask, (64, 64), step=64)

        img_patch_shape = img_patches[:, :, :, :, :, :1]
        img_patch_shape = np.squeeze(img_patch_shape)
        img_shape = crop_img[:, :, :1]
        img_shape = np.squeeze(img_shape)

        image_array = np.concatenate(img_patches, axis=0)
        image_array = np.squeeze(image_array, axis=1)
        img = preprocess_input(image_array)

        mask_array = np.concatenate(mask_patches, axis=0)

        # Perform prediction
        test_pred = model.predict(img)
        test_pred_argmax = np.argmax(test_pred, axis=3)

        test_pred_argmax_reshaped = test_pred_argmax.reshape(img_patch_shape.shape)

        reconstructed_image = unpatchify(test_pred_argmax_reshaped, img_shape.shape)

        # Calculate overall priority
        total_pixels = reconstructed_image.size
        total_pixels_original = crop_mask.size

        class_counts = np.bincount(reconstructed_image.flatten(), minlength=len(class_priorities))
        class_original_counts = np.bincount(crop_mask.flatten(), minlength=len(class_priorities))

        class_percentages = class_counts / total_pixels * 100
        class_percentages_original = class_original_counts / total_pixels_original * 100

        overall_priority = sum(class_percentages[i] * class_priorities[class_label] for i, class_label in enumerate(class_priorities))
        overall_priority_original = sum(class_percentages_original[i] * class_priorities[class_label] for i, class_label in enumerate(class_priorities))

        # Append data to lists and arrays
        image_numbers.append(img_num)
        overall_priorities.append(overall_priority)
        overall_priorities_original.append(overall_priority_original)
        crop_images.append(crop_img)
        crop_masks.append(crop_mask)
        reconstructed_images.append(reconstructed_image)

        _min, _max = np.amin(test_pred_argmax), np.amax(test_pred_argmax)

    # Combine and sort data based on overall priorities
    combined_data = list(zip(image_numbers, overall_priorities, overall_priorities_original, crop_images, crop_masks, reconstructed_images))
    sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)

    # Display the sorted results
    for img_num, priority, original_prio, crop_img, crop_mask, reconstructed_image in sorted_data:
        print(f"Image {img_num} - Overall Priority: {priority} and Ground Truth Priority: {original_prio}")

        plt.figure(figsize=(15, 15), dpi=80)

        plt.subplot(1, 3, 1)
        plt.subplots_adjust(hspace=0)
        plt.subplots_adjust(wspace=0)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(crop_img.astype('uint8'))
        plt.title("Original Image")

        plt.subplot(1, 3, 2)
        show_mask_with_label(crop_mask, labels, _min, _max, alpha=0.9,ticks=False)
        plt.title("Ground Truth Mask")

        plt.subplot(1, 3, 3)
        show_mask_with_label(reconstructed_image, labels, _min, _max, alpha=0.9, legend=True, ticks=False)
        plt.title("Predicted Mask")

        plt.show()
