"""
We can't load MNIST dataset on the browser.
We use this script to export MNIST dataset images to folders and we put these folders in the frontend/public folder.
We don't export the entire dataset, only 30 images per digit from 1 to 9.
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST dataset
def load_mnist():
    # Load MNIST dataset using TensorFlow (it will download if not already present)
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    return train_images, train_labels

# Save images to a folder
def save_images_by_digit(images, labels, digit, folder_path, num_images=30):
    # Filter images for the given digit
    digit_images = images[labels == digit]
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the first 'num_images' images of the specified digit
    for i in range(min(num_images, len(digit_images))):
        image = digit_images[i]
        # Save the image as a .png file
        image_path = os.path.join(folder_path, f"{digit}_{i+1}.png")
        plt.imsave(image_path, image, cmap='gray')

# Main function
def main():
    # Load dataset
    train_images, train_labels = load_mnist()
    
    # Set the path where the folders should be located (frontend/public)
    output_directory = os.path.join('..', 'public', 'mnist')
    
    # Loop through digits 1 to 9 and save 30 images per digit
    for digit in range(1, 10):
        folder_path = os.path.join(output_directory, str(digit))  # Folder inside frontend/public
        save_images_by_digit(train_images, train_labels, digit, folder_path)

    print("Images have been saved to:", output_directory)

if __name__ == "__main__":
    main()
