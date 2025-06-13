import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (64, 64)
DATA_DIR = "."
CLASSES = ["stressed", "depressed", "normal"]
EPOCHS = 300
BATCH_SIZE = 32
SYNTHETIC_PER_CLASS = 300


# 1. Preprocessing Function
def load_images(data_dir):
    X, y = [], []
    for idx, label in enumerate(CLASSES):
        folder = os.path.join(data_dir, label)
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                img = img / 255.0
                X.append(img)
                y.append(idx)
    return np.array(X), tf.keras.utils.to_categorical(y, num_classes=len(CLASSES))


X, y = load_images(DATA_DIR)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


# 2. GAN Model (DCGAN)
def build_generator():
    model = models.Sequential([
        layers.Input(shape=(100,)),  # ← updated as per best practices
        layers.Dense(8 * 8 * 128, activation="relu"),
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation='relu'),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation='relu'),
        layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", activation='sigmoid')
    ])
    return model


def build_discriminator():
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),  # ← updated as per best practices
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid")
    ])
    return model


def train_gan(real_images, epochs=EPOCHS):
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(optimizer='adam', loss='binary_crossentropy')

    noise_input = layers.Input(shape=(100,))
    generated_image = generator(noise_input)
    discriminator.trainable = False
    valid = discriminator(generated_image)
    combined = models.Model(noise_input, valid)
    combined.compile(optimizer='adam', loss='binary_crossentropy')

    for epoch in range(epochs):
        idx = np.random.randint(0, real_images.shape[0], BATCH_SIZE)
        real = real_images[idx]
        noise = np.random.normal(0, 1, (BATCH_SIZE, 100))
        fake = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real, np.ones((BATCH_SIZE, 1)))
        d_loss_fake = discriminator.train_on_batch(fake, np.zeros((BATCH_SIZE, 1)))

        noise = np.random.normal(0, 1, (BATCH_SIZE, 100))
        g_loss = combined.train_on_batch(noise, np.ones((BATCH_SIZE, 1)))

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - D loss: {d_loss_real:.4f}, G loss: {g_loss:.4f}")  # ← FIXED

    return generator


# 3. GAN Augmentation
augmented_images, augmented_labels = [], []

for i in range(len(CLASSES)):
    print(f"\nTraining GAN for class: {CLASSES[i]}")
    class_images = X_train[np.argmax(y_train, axis=1) == i]
    gen = train_gan(class_images)

    noise = np.random.normal(0, 1, (SYNTHETIC_PER_CLASS, 100))
    fake_imgs = gen.predict(noise)
    augmented_images.extend(fake_imgs)
    augmented_labels.extend([i] * SYNTHETIC_PER_CLASS)

X_aug = np.array(augmented_images)
y_aug = tf.keras.utils.to_categorical(np.array(augmented_labels), num_classes=len(CLASSES))

# Combine real and synthetic data
X_train_final = np.concatenate((X_train, X_aug), axis=0)
y_train_final = np.concatenate((y_train, y_aug), axis=0)


# 4. CNN Classifier
def build_classifier():
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),  # ← added Input layer as per warning
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


classifier = build_classifier()
classifier.fit(X_train_final, y_train_final, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# 5. Evaluation
y_pred = classifier.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=CLASSES))

cm = confusion_matrix(y_true_classes, y_pred_classes)
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
