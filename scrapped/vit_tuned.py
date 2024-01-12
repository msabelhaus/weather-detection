import tensorflow as tf
import os
# from tensorflow.keras.optimizers import Adam
# from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, Adamax
from vit_keras import vit

# Set the paths to your preprocessed dataset
train_data_dir = os.path.abspath("../../data/processed/train")
validation_data_dir = os.path.abspath("../../data/processed/test")

# Batch size and image dimensions
batch_size = 32
input_shape = (256, 256, 3)  # Set your desired target size

# Image data generator for training
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical'
)

# Image data generator for validation
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical'
)

# # Display an image from the generator
# sample_images, sample_labels = next(train_generator)
# sample_image = sample_images[0]  # Assuming you want to display the first image in the batch

# # Save the image to the specified directory
# save_dir = os.path.abspath("../../images")
# image_filename = os.path.join(save_dir, "sample_image.png")
# plt.imshow(sample_image)
# plt.title(f"Image Shape: {sample_image.shape}")
# plt.savefig(image_filename)
# plt.close()

# Display and save the first 10 images from the generator
# num_images_to_save = 10
save_dir = os.path.abspath("../../images")
# for i in range(num_images_to_save):
#     sample_images, sample_labels = next(train_generator)
#     sample_image = sample_images[0]  # Assuming you want to save the first image in the batch
#     image_filename = os.path.join(save_dir, os.path.basename(train_generator.filenames[i]))
    
#     plt.imshow(sample_image)
#     plt.title(f"Image Shape: {sample_image.shape}")
#     plt.savefig(image_filename)
#     plt.close()

# # Callbacks
# reduce_learning_rate = ReduceLROnPlateau(
#     monitor='val_loss', factor=0.25, patience=5, verbose=1, mode='auto',
#     min_delta=1e-10, cooldown=0, min_lr=0
# )

# early_stopping = EarlyStopping(
#     monitor='val_loss', min_delta=0, patience=9, verbose=1, mode='auto',
#     baseline=None, restore_best_weights=True
# )
# callbacks = [reduce_learning_rate, early_stopping]

# Number of classes
num_classes = len(train_generator.class_indices)

# Model
image_size = 256
epochs = 10

vit_model = vit.vit_b16(
    image_size=image_size,
    activation='softmax',
    pretrained=True,
    include_top=True,
    pretrained_top=False,
    classes=num_classes
)

model = tf.keras.Sequential([
        vit_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
        tf.keras.layers.Dense(11, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
                    bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
        tf.keras.layers.BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
        Dropout(rate= 0.45, seed= 123),
        tf.keras.layers.Dense(5, 'softmax')
    ],
    name = 'vision_transformer')

model.summary()

model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

history = model.fit(train_generator,
                    epochs=epochs,
                    steps_per_epoch=train_generator.samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // batch_size,
                    verbose=1)

# Plot validation set loss
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(save_dir, 'validation_loss_plot_vit_tuned.png'))
# plt.show()