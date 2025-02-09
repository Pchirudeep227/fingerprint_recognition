import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # Assuming 10 classes for fingerprint recognition
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation data
train_generator = train_datagen.flow_from_directory('data/train', target_size=(64, 64), color_mode='grayscale', batch_size=32, class_mode='categorical')
validation_generator = test_datagen.flow_from_directory('data/validation', target_size=(64, 64), color_mode='grayscale', batch_size=32, class_mode='categorical')

# Train the model
model.fit(train_generator, steps_per_epoch=8000//32, epochs=25, validation_data=validation_generator, validation_steps=2000//32)

model.save('fingerprint_recognition_model.h5')