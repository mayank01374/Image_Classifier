import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


folders = {
    "stressed": "stressed",
    "depressed": "depressed",
    "normal": "normal"
}


img_size = (64, 64)
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

def extract_hog_features(image):
    return hog(image,
               orientations=orientations,
               pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block,
               block_norm='L2-Hys',
               transform_sqrt=True,
               feature_vector=True)

def load_dataset():
    X, y = [], []
    for label_name, folder_path in folders.items():
        print(f"Loading images from: {folder_path}")
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Skipping unreadable file: {file}")
                continue
            img = cv2.resize(img, img_size)
            features = extract_hog_features(img)
            X.append(features)
            y.append(label_name)  # Use class name directly
    return np.array(X), np.array(y)

def train_and_evaluate_knn(X, y):
    class_names = sorted(list(set(y)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn_clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')  # You can tune these
    knn_clf.fit(X_train, y_train)

    y_pred = knn_clf.predict(X_test)

    print("\n--- KNN Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names))

    display_confusion_matrix(y_test, y_pred, class_names)

def display_confusion_matrix(y_true, y_pred, labels, title="KNN Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    X, y = load_dataset()
    print(f"\nDataset loaded. Total samples: {len(X)}")
    train_and_evaluate_knn(X, y)
