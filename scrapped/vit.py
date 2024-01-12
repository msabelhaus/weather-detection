import tensorflow as tf
import pickle
import os
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from vit_keras import vit, utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


# Set the paths to your preprocessed dataset
train_data_dir = os.path.abspath("../../data/processed/train")
validation_data_dir = os.path.abspath("../../data/processed/test")

# Image data generators for training and validation
# train_datagen = ImageDataGenerator()
train_datagen = ImageDataGenerator()

# validation_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

# Batch size and image dimensions
batch_size = 32
# batch_size = 100
input_shape = (256, 256, 3)

# Flow training images in batches using the generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Callbacks
reduce_learning_rate = ReduceLROnPlateau(
    monitor='val_loss', factor=0.25, patience=5, verbose=1, mode='auto',
    min_delta=1e-10, cooldown=0, min_lr=0
)

early_stopping = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=9, verbose=1, mode='auto',
    baseline=None, restore_best_weights=True
)
callbacks = [reduce_learning_rate, early_stopping]


# Number of classes
num_classes = len(train_generator.class_indices)

# Model
# image_size = 256
# model = vit.vit_b16(
#     image_size = image_size,
#     activation = 'softmax',
#     pretrained = True,
#     include_top = True,
#     pretrained_top = False,
#     classes = num_classes
# )

image_size = 256
epochs = 5

model = vit.vit_b16(
    image_size = image_size,
    activation = 'softmax',
    pretrained = True,
    include_top = True,
    pretrained_top = False,
    classes = num_classes
)

# model.compile(optimizer=Adam(lr=0.0001, decay=1e-6), loss='binary_crossentropy', metrics=['accuracy'])
# history = model.fit(train_generator, epochs=5, validation_data=validation_generator, verbose=1)

model.compile(optimizer=Adam(lr=0.001, decay=1e-6),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,
                    epochs=epochs,
                    steps_per_epoch=train_generator.samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // batch_size,
                    callbacks=callbacks,
                    verbose=1)

# # Number of classes
# num_classes = len(train_generator.class_indices)

# # Create the CNN model
# cnn_model = create_cnn_model(input_shape=input_shape, num_classes=num_classes)

# # Train the model
# epochs = 10
# history = cnn_model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size
# )


# # Save the model using Pickle
# if not os.path.exists(os.path.abspath("../models")):
#     os.makedirs(os.path.abspath("../models"))    

# pickle_path = os.path.abspath("../models/cnn.pkl")
# with open(pickle_path, 'wb') as file:
#     pickle.dump(cnn_model, file)