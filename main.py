import os
import numpy as np
from PIL import Image  # Sử dụng Pillow thay cho OpenCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import random

# Đường dẫn đến thư mục chứa ảnh
data_path = r"C:\Users\Admin\Downloads\anh"

# Danh sách để lưu ảnh và nhãn
images = []
labels = []

# Đọc ảnh từ thư mục và gán nhãn
for image_file in os.listdir(data_path):
    img_path = os.path.join(data_path, image_file)

    # Kiểm tra nếu file là ảnh (có đuôi jpg, png, v.v.)
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        img = Image.open(img_path)

        # Thay đổi kích thước ảnh
        img_resized = img.resize((128, 128))  # Thay đổi kích thước ảnh
        images.append(np.array(img_resized))  # Chuyển đổi ảnh sang mảng numpy

        # Gán nhãn theo tên file (bạn có thể thay đổi cách gán nhãn)
        label = image_file.split('.')[0]  # Lấy phần tên file trước dấu chấm
        labels.append(label)

# Chuyển đổi thành mảng numpy
images = np.array(images)
labels = np.array(labels)

# Hàm tăng cường dữ liệu
def augment_data(images, labels):
    augmented_images = []
    augmented_labels = []

    for img_array, label in zip(images, labels):
        img = Image.fromarray(img_array)  # Chuyển đổi từ mảng numpy sang đối tượng PIL.Image

        # Thêm ảnh gốc
        augmented_images.append(np.array(img))
        augmented_labels.append(label)

        # Tăng cường dữ liệu: Xoay ảnh
        angles = [0, 90, 180, 270]
        angle = random.choice(angles)
        img_rotated = img.rotate(angle)  # Xoay ảnh
        augmented_images.append(np.array(img_rotated))  # Chuyển đổi sang mảng numpy
        augmented_labels.append(label)

        # Tăng cường dữ liệu: Lật ngang
        img_flipped = img.transpose(method=Image.FLIP_LEFT_RIGHT)  # Lật ngang
        augmented_images.append(np.array(img_flipped))  # Chuyển đổi sang mảng numpy
        augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)

# Tăng cường dữ liệu
images, labels = augment_data(images, labels)

# Chuyển đổi ảnh thành vector
images = images.reshape(images.shape[0], -1)

# Chia dữ liệu thành các bộ train-test
splits = [(0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.4, 0.6)]
results = {}

for train_size, test_size in splits:
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(images, labels, train_size=train_size, random_state=42)

    # Huấn luyện mô hình SVM
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)

    # Huấn luyện mô hình KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_predictions)

    # Lưu kết quả
    results[(train_size, test_size)] = {
        'SVM Accuracy': svm_accuracy,
        'KNN Accuracy': knn_accuracy,
        'SVM Predictions': svm_predictions,
        'KNN Predictions': knn_predictions,
        'y_test': y_test
    }

# Hiển thị kết quả
for split, metrics in results.items():
    print(f"Train-Test Split: {int(split[0] * 100)}-{int(split[1] * 100)}")
    print(f"SVM Accuracy: {metrics['SVM Accuracy']:.2f}")
    print(f"KNN Accuracy: {metrics['KNN Accuracy']:.2f}\n")

    # In báo cáo phân loại với zero_division
    print("SVM Classification Report:")
    print(classification_report(metrics['y_test'], metrics['SVM Predictions'], zero_division=0))

    print("KNN Classification Report:")
    print(classification_report(metrics['y_test'], metrics['KNN Predictions'], zero_division=0))
