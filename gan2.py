import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
import tensorflow as tf
from tensorflow.keras import layers, models, preprocessing, applications
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


np.random.seed(42)
tf.random.set_seed(42)


IMAGE_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS_GAN = 100
EPOCHS_CLASSIFIER = 50
LATENT_DIM = 256
NUM_CLASSES = 3


script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = script_dir
folders = ['Stressed', 'Depressed', 'Normal']


def load_and_preprocess_data():
    images = []
    labels = []

    for label, folder in enumerate(folders):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder '{folder}' not found in {data_dir}")

        print(f"Loading images from {folder}...")
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


        for i, file in enumerate(files):
            if i % 500 == 0:
                print(f"Processed {i}/{len(files)} images from {folder}")

            img_path = os.path.join(folder_path, file)
            try:
                img = preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
                img_array = preprocessing.image.img_to_array(img)
                img_array = (img_array - 127.5) / 127.5
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Skipping {file}: {str(e)}")
                continue

    images = np.array(images)
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=NUM_CLASSES)

    print(f"\nSuccessfully loaded {len(images)} images total")
    print(f"Class distribution: {np.sum(labels, axis=0)}")

    return images, labels



print("Loading and preprocessing images...")
images, labels = load_and_preprocess_data()


X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)



def build_generator(latent_dim):
    model = models.Sequential()


    model.add(layers.Dense(256 * 16 * 16, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((16, 16, 256)))


    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))


    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))


    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))


    model.add(layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model


def build_discriminator(img_shape):
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


def train_gan(generator, discriminator, gan, X_train, epochs, batch_size, latent_dim):
    batch_count = X_train.shape[0] // batch_size

    for epoch in range(epochs):
        for _ in range(batch_count):

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_imgs = X_train[idx]


            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_imgs = generator.predict(noise, verbose=0)


            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))


            d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = gan.train_on_batch(noise, real_labels)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

    return generator



print("\nBuilding GAN components...")
img_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                      metrics=['accuracy'])

generator = build_generator(LATENT_DIM)

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

print("Training GAN...")
generator = train_gan(generator, discriminator, gan, X_train, EPOCHS_GAN, BATCH_SIZE, LATENT_DIM)



def build_classifier(input_shape, num_classes):
    base_model = applications.EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape)


    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model



print("\nBuilding classifier...")
classifier = build_classifier(img_shape, NUM_CLASSES)
classifier.compile(optimizer=Adam(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True)

print("Training classifier...")
history = classifier.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_CLASSIFIER,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping])



def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)


    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')


    class_report = classification_report(y_true_classes, y_pred_classes, target_names=folders)

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\n=== Classification Report ===")
    print(class_report)


    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=folders, yticklabels=folders)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return accuracy, precision, recall, f1


print("\nFinal Evaluation on Test Set:")
test_loss, test_acc = classifier.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

evaluate_model(classifier, X_test, y_test)


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()