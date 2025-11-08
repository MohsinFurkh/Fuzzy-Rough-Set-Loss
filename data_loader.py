import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

def load_dataset(data_dir, img_size=(128, 128), test_size=0.2, val_size=0.1, random_state=42):
    """
    Load and preprocess the BUSI dataset.
    
    Args:
        data_dir: Path to the BUSI dataset directory
        img_size: Target image size (height, width)
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')
    
    # Get list of image files (excluding mask files)
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    
    # Initialize lists to store images and masks
    images = []
    masks = []
    
    # Load and preprocess images and masks
    for img_file in image_files:
        # Load image
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        img = cv2.resize(img, img_size[::-1])  # cv2 uses (width, height)
        img = img.astype(np.float32) / 255.0
        
        # Load corresponding mask - same filename in masks directory
        mask_path = os.path.join(masks_dir, img_file)
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {img_file}")
            continue
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask {mask_path}")
            continue
            
        mask = cv2.resize(mask, img_size[::-1], interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)  # Binarize mask
        
        images.append(img)
        masks.append(mask[..., np.newaxis])  # Add channel dimension
    
    # Convert to numpy arrays
    images = np.array(images)
    masks = np.array(masks)
    
    # Split into train, validation, and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        images, masks, test_size=test_size, random_state=random_state
    )
    
    # Further split training data into train and validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def visualize_dataset_samples(datasets, num_samples=2):
    """
    Visualize random samples from train, validation, and test sets.
    
    Args:
        datasets: Tuple containing ((x_train, y_train), (x_val, y_val), (x_test, y_test))
        num_samples: Number of samples to display from each set
    """
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = datasets
    
    # Create a figure with subplots
    fig, axes = plt.subplots(3, num_samples * 2, figsize=(15, 10))
    fig.suptitle('Dataset Samples (Image | Mask)', fontsize=16)
    
    # Function to plot samples for a given dataset
    def plot_samples(images, masks, row_idx, title):
        # Get random indices
        if len(images) < num_samples:
            print(f"Warning: Not enough samples in {title} set to show {num_samples} samples")
            return
            
        indices = np.random.choice(len(images), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            # Plot image
            ax = axes[row_idx, 2*i]
            ax.imshow(images[idx])
            ax.set_title(f"{title} Image")
            ax.axis('off')
            
            # Plot mask
            ax = axes[row_idx, 2*i + 1]
            ax.imshow(masks[idx].squeeze(), cmap='gray')
            ax.set_title(f"{title} Mask")
            ax.axis('off')
    
    # Plot training set samples
    plot_samples(x_train, y_train, 0, 'Train')
    
    # Plot validation set samples
    plot_samples(x_val, y_val, 1, 'Validation')
    
    # Plot test set samples
    plot_samples(x_test, y_test, 2, 'Test')
    
    plt.tight_layout()
    plt.show()
    
    # Also print dataset statistics
    print("\nDataset Statistics:")
    print(f"Training set: {len(x_train)} samples")
    print(f"Validation set: {len(x_val)} samples")
    print(f"Test set: {len(x_test)} samples")
    print(f"Image shape: {x_train[0].shape}")
    print(f"Mask shape: {y_train[0].shape}")
    print(f"Mask values (min, max): ({y_train[0].min()}, {y_train[0].max()})")


def create_dataset(x, y, batch_size=8, shuffle=True, augment=False):
    """
    Create a TensorFlow dataset with optional data augmentation.
    
    Args:
        x: Input images (numpy array)
        y: Target masks (numpy array)
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        augment: Whether to apply data augmentation
        
    Returns:
        TensorFlow dataset
    """
    def preprocess(image, mask):
        # Ensure image and mask have the correct shape and type
        image = tf.cast(image, tf.float32)
        mask = tf.cast(mask[..., 0], tf.float32)  # Remove channel dim from mask if present
        
        # Add channel dimension if not present (for grayscale images)
        if len(tf.shape(image)) == 2:
            image = tf.expand_dims(image, axis=-1)
        if len(tf.shape(mask)) == 2:
            mask = tf.expand_dims(mask, axis=-1)
            
        return image, mask
    
    def augment_data(image, mask):
        # Apply the same random transformations to both image and mask
        # Convert to float32 for augmentation
        image = tf.cast(image, tf.float32)
        mask = tf.cast(mask, tf.float32)
        
        # Random horizontal flip (50% chance)
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
            
        # Random vertical flip (50% chance)
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
            
        # Random rotation (0, 90, 180, or 270 degrees)
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        mask = tf.image.rot90(mask, k=k)
        
        # Random brightness (only for image, not mask)
        image = tf.image.random_brightness(image, max_delta=0.1)
        
        # Ensure values are still in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        mask = tf.clip_by_value(mask, 0.0, 1.0)
        
        return image, mask
    
    # Create dataset from numpy arrays
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    # Apply preprocessing
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Cache the dataset to disk to speed up training
    dataset = dataset.cache()
    
    # Shuffle if needed
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    # Apply augmentation if needed
    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch the data
    dataset = dataset.batch(batch_size)
    
    # Prefetch for better performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
