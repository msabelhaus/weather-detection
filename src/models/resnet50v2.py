import tensorflow as tf
import pickle
import os
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, Adamax
import numpy as np

# Set the paths to your preprocessed dataset
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
    target_size=input_shape[:2],
    # target_size=input_shape,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=input_shape[:2],
    # target_size=input_shape,
    batch_size=batch_size,
    class_mode='categorical'
)

# Number of classes
num_classes = len(train_generator.class_indices)

# Load efficientnetb3 pretrained model
resnet_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top= False, weights= "imagenet", input_shape= input_shape, pooling= 'max')

model = Sequential([
    resnet_model,
    Flatten(),
    # BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
    Dense(256, kernel_regularizer= regularizers.l2(0.016), activity_regularizer= regularizers.l1(0.006),
                bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
    # Dense(128, activation='relu'),
    Dropout(rate= 0.45, seed= 123),
    Dense(num_classes, activation= 'softmax')
])

model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])
# model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics= ['accuracy'])

model.summary()

epochs = 10   # number of all epochs in training

# history = model.fit(x= train_generator, epochs= epochs, verbose= 1, validation_data= validation_generator, 
#                     validation_steps= None, shuffle= False)
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    verbose = 1)

# Plot validation set loss
save_dir = os.path.abspath("../../images")
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(save_dir, 'validation_loss_plot_resnet_adamx.png'))
plt.show()

# Save the model using Pickle
if not os.path.exists(os.path.abspath("../models")):
    os.makedirs(os.path.abspath("../models"))    

pickle_path = os.path.abspath("../models/resnet50v2.pkl")
with open(pickle_path, 'wb') as file:
    pickle.dump(model, file)
