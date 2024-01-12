import tensorflow as tf
<<<<<<< HEAD
import pickle
import os
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the CNN model
def create_cnn_model(input_shape=(128, 128, 3), num_classes=10):
# def create_cnn_model(input_shape=(256, 256, 3), num_classes=10):
=======
import os
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the CNN model
def create_cnn_model(input_shape=(256, 256, 3), num_classes=10):
>>>>>>> 6f6bc9f0106b5322679811e8c657083b28b0c7de
    model = models.Sequential()
    
    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model

<<<<<<< HEAD
# Set file paths
=======
# Set the paths to your preprocessed dataset
>>>>>>> 6f6bc9f0106b5322679811e8c657083b28b0c7de
train_data_dir = os.path.abspath("../../data/processed/train")
validation_data_dir = os.path.abspath("../../data/processed/test")

# Image data generators for training and validation
train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

# Batch size and image dimensions
batch_size = 32
input_shape = (256, 256, 3)

# Flow training images in batches using the generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
<<<<<<< HEAD
=======
    target_size=input_shape[:2],
>>>>>>> 6f6bc9f0106b5322679811e8c657083b28b0c7de
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Number of classes
num_classes = len(train_generator.class_indices)

# Create the CNN model
cnn_model = create_cnn_model(input_shape=input_shape, num_classes=num_classes)

<<<<<<< HEAD
# Model fitting
=======
# Train the model
>>>>>>> 6f6bc9f0106b5322679811e8c657083b28b0c7de
epochs = 10
history = cnn_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

<<<<<<< HEAD
# Plot validation set loss & save image
save_dir = os.path.abspath("../../images")
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(save_dir, 'validation_loss_plot_cnn.png'))
plt.show()
=======
>>>>>>> 6f6bc9f0106b5322679811e8c657083b28b0c7de

# Save the model using Pickle
if not os.path.exists(os.path.abspath("../models")):
    os.makedirs(os.path.abspath("../models"))    

pickle_path = os.path.abspath("../models/cnn.pkl")
with open(pickle_path, 'wb') as file:
    pickle.dump(cnn_model, file)
<<<<<<< HEAD
    
    
    
=======


>>>>>>> 6f6bc9f0106b5322679811e8c657083b28b0c7de
