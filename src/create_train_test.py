import os
import random
import shutil

path = "../data/raw/dataset/" # Assign root directory

# Initalize empty list to fill with each class name
weather_classes = []

for dirpath, dirnames, filenames in os.walk(path):
    directory_level = dirpath.replace(path, "")
    if directory_level:  # Exclude empty strings
        weather_classes.append(directory_level)

# Make train and test directories for all weather classes
def make_dirs(group):
    for wc in weather_classes:
        os.makedirs(os.path.join(path, os.pardir, group, wc), exist_ok=True)

try:
    # Attempt to create directories
    make_dirs("train")
    make_dirs("test")
except OSError as e:
    print(f"Error creating directories: {e}")
    
# Set test percentage
test_percentage = 20

# Loop through folders, randomly select 20% for the test set, and move files to the appropriate location
for wc in weather_classes:
    wc_path = os.path.join(path, wc)
    
    # Get list of all images
    images = [f for f in os.listdir(wc_path) if f.endswith(".jpg")]
    
    # Randomly select images for test set
    num_test_images = int(len(images) * test_percentage / 100)
    test_images = random.sample(images, num_test_images)
    
    # Move the selected test images to the test set directory
    for image in test_images:
        source_path = os.path.join(wc_path, image)
        target_path = os.path.join(path, os.pardir, "test", wc, image)
        shutil.move(source_path, target_path)

    # Move the remaining images to the train set directory
    for image in [im for im in images if im not in test_images]:
        source_path = os.path.join(wc_path, image)
        target_path = os.path.join(path, os.pardir, "train", wc, image)
        shutil.move(source_path, target_path)
    
print("Data split into train and test sets.")