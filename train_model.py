import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split


# === Configuration ===
DATA_DIR = "plantVillage" 
IMG_SIZE = 64

# === Data Loading ===
print("[INFO] Loading images...")
x, y = [], []

for label in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir):
        continue
    
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARNING] Could not read image {img_path}, skipping...")
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()  # Normalize and flatten the histogram
            
            x.append(hist)
            y.append(label)
        except Exception as e:
            print(f"[ERROR] Could not process image {img_path}: {e}")
            
x = np.array(x)
y = np.array(y)


from sklearn.preprocessing import LabelEncoder # encode labels as integers
print("[INFO] Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

 # == Add PCA ===
from sklearn.decomposition import PCA


print("[INFO] Applying PCA...")
pca = PCA(n_components=512)
x_pca = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_pca, y_encoded, test_size=0.2, random_state=42)

joblib.dump(pca, "models/pca.pkl")

# === Train/Test Split ===

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

# === Train Model ===
from sklearn.ensemble import RandomForestClassifier
print("[INFO] Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# === Evaluate Model ===
from sklearn.metrics import classification_report
y_pred = model.predict(x_test)
print("[INFO] Model evaluation:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))



# === Save Model ===

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("[INFO] Model and label encoder saved.")

import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import confusion_matrix

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()