import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

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
    for label_idx, (label_name, folder_path) in enumerate(folders.items()):
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
            y.append(label_idx)
    return np.array(X), np.array(y)


def display_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()


def train_and_evaluate(X, y):
    class_names = list(folders.keys())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    svm_clf = SVC(kernel='linear')
    svm_clf.fit(X_train, y_train)
    y_pred_svm = svm_clf.predict(X_test)
    print("\n--- SVM Classification Report ---")
    print(classification_report(y_test, y_pred_svm, target_names=class_names))
    print("SVM Confusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_svm),
                       index=class_names,
                       columns=class_names))
    display_confusion_matrix(y_test, y_pred_svm, class_names, title="SVM Confusion Matrix")
    print("\n")


    lda_clf = LinearDiscriminantAnalysis()
    lda_clf.fit(X_train, y_train)
    y_pred_lda = lda_clf.predict(X_test)
    print("\n--- LDA Classification Report ---")
    print(classification_report(y_test, y_pred_lda, target_names=class_names))
    print("LDA Confusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_lda),
                       index=class_names,
                       columns=class_names))
    display_confusion_matrix(y_test, y_pred_lda, class_names, title="LDA Confusion Matrix")


if __name__ == "__main__":
    X, y = load_dataset()
    print(f"Dataset loaded. Total samples: {len(X)}")
    train_and_evaluate(X, y)