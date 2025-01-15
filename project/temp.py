import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for

# Flask Uygulaması Başlatma
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 1. Veri Yolu ve Görsellerin Boyutlandırılması
img_size = (64, 64)  # Görseller 64x64 boyutunda yeniden boyutlandırılacak.
dataset_folder = "dataset"  # Görsellerin bulunduğu ana klasör.

# Veri ve Etiketler için Boş Listeler
data = []
labels = []

# 2. Görsellerin İşlenmesi ve Öznitelik Çıkarımı
for class_name in os.listdir(dataset_folder):
    class_path = os.path.join(dataset_folder, class_name)
    if os.path.isdir(class_path):  # Sadece klasörleri işlemeye devam et
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Görselleri gri tonlamaya dönüştür
                if img is not None:
                    img_resized = cv2.resize(img, img_size)  # Görselleri yeniden boyutlandır
                    hog_features = hog(
                        img_resized,
                        orientations=9,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),
                        block_norm='L2-Hys',
                        transform_sqrt=True
                    )
                    data.append(hog_features) 
                    labels.append(int(class_name))  # Etiketi ekle
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

# 3. Verilerin Eğitim ve Test Setlerine Bölünmesi
data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.3, random_state=42
)

# 4. Random Forest Modeli
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

rf_train_accuracy = accuracy_score(y_train, rf_train_pred)
rf_test_accuracy = accuracy_score(y_test, rf_test_pred)

# 5. SVM Modeli
svm_model = SVC(
    kernel='rbf',
    C=0.5,
    gamma='scale',
    random_state=42
)
svm_model.fit(X_train, y_train)

svm_train_pred = svm_model.predict(X_train)
svm_test_pred = svm_model.predict(X_test)

svm_train_accuracy = accuracy_score(y_train, svm_train_pred)
svm_test_accuracy = accuracy_score(y_test, svm_test_pred)

# 6. En İyi Modelin Seçimi
if rf_test_accuracy > svm_test_accuracy:
    best_model_name = "Random Forest"
    best_model = rf_model
    best_test_accuracy = rf_test_accuracy
else:
    best_model_name = "SVM"
    best_model = svm_model
    best_test_accuracy = svm_test_accuracy

print(f"Best Model: {best_model_name}")
print(f"Best Model Test Accuracy: {best_test_accuracy:.4f}")

# Flask Rotası: Ana Sayfa
@app.route('/')
def index():
    return render_template('index.html')

# Flask Rotası: Tahmin Yapma
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Görseli İşleme ve Öznitelik Çıkarma
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, img_size)
        hog_features = hog(
            img_resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True
        ).reshape(1, -1)

        # Tahmin Yapma
        predicted_class = best_model.predict(hog_features)[0]

        return render_template('result.html',
                               uploaded_image=file.filename,
                               predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
