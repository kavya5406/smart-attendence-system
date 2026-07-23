import os
import cv2
import joblib

import mlflow
import mlflow.sklearn

from mtcnn import MTCNN
from keras_facenet import FaceNet

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# Dataset Path
# -------------------------
DATASET_PATH = "dataset/Image_Train"
MODEL_PATH = "models/face_recognition_model.pkl"

# -------------------------
# Load Models
# -------------------------
detector = MTCNN()
embedder = FaceNet()

embeddings = []
labels = []

print("Loading Dataset...\n")

# -------------------------
# Read Dataset
# -------------------------
for person in os.listdir(DATASET_PATH):

    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    print(f"Reading {person}")

    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = detector.detect_faces(image)

        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]['box']

        x = max(0, x)
        y = max(0, y)

        face = image[y:y+h, x:x+w]

        try:
            face = cv2.resize(face, (160, 160))
        except:
            continue

        embedding = embedder.embeddings([face])[0]

        embeddings.append(embedding)
        labels.append(person)

print("\n-----------------------------------")
print("Total Images Used :", len(labels))
print("-----------------------------------")

# -------------------------
# Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    embeddings,
    labels,
    test_size=0.2,
    random_state=42,
    
)

# -------------------------
# Train Model
# -------------------------
print("\nTraining SVM Model...\n")

model = SVC(kernel='linear', probability=True)

model.fit(X_train, y_train)

# -------------------------
# Prediction
# -------------------------
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("-----------------------------------")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("-----------------------------------")

print("\nClassification Report\n")
print(classification_report(y_test, predictions))

# -------------------------
# Save Model
# -------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)


