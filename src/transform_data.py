import os
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np

training_dir = os.path.abspath("../data/raw/train")
test_dir = os.path.abspath("../data/raw/test")

load_training_data = tf.keras.utils.image_dataset_from_directory(
    # Note: default batch_size=32, image_size=(256,256)
    training_dir,
    shuffle=True,
    seed=42,
    labels = 'inferred', # Labels defiend by subdirectories
    label_mode='categorical',
    # validation_split=0.1,  # 10% split of the training data for validation
    # subset='training',  # Specifying 'training' to get the training split
)

load_test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    shuffle=True,
    seed=42,
    labels = 'inferred',
    label_mode='categorical'
)

# Load some example images
print("4 sample rainbow images:")
fig=plt.figure(figsize=(16, 16))
for i in range(1,5):
    folder = os.path.join(training_dir, 'rainbow')
    random_img = random.choice(os.listdir(folder))
    img = np.array(Image.open(folder+'/'+random_img))
    fig.add_subplot(1, 4, i)
    plt.imshow(img)
    plt.axis('off')
plt.show()

def preprocess_images(input_dir, output_dir, size, pixels):
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    
    for category in os.listdir(training_dir):
        category_path = os.path.join(training_dir, category)
        
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            
            image = Image.open(file_path)
            
            # Resize the image
            image = image.resize(size, Image.ANTIALIAS)
            image = image.convert("RGB") # Need to convert to RGB to save as .jpeg
            
            # Rescale the image to [-1,1]
            image_array = np.array(image)
            image_array = (image_array / 255.0) * (pixels[1] - pixels[0]) + pixels[0]
            processed_image = Image.fromarray((image_array * 255).astype(np.uint8))
            
            # Save the processed image to the output directory
            output_path = os.path.join(output_dir, category, filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            processed_image.save(output_path)

output_train_dir = os.path.abspath("../data/processed/test")     
output_test_dir = os.path.abspath("../data/processed/train")             
preprocess_images(test_dir, output_test_dir, size=(256,256), pixels=(-1,1))
preprocess_images(training_dir, output_train_dir, size=(256,256), pixels=(-1,1))