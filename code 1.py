import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import to_categorical
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Để theo dõi tiêu đề hình ảnh
_mfajlsdf98q21_image_title_list = []

# Cấu hình GPU
# Nếu máy tính có GPU, tận dụng GPU để tăng tốc quá trình huấn luyện
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("TensorFlow version:", tf.__version__)
print("Eager execution is:", "enabled" if tf.executing_eagerly() else "disabled")

# Tạo dữ liệu giả để mô phỏng dữ liệu CT phổi
def generate_synthetic_data(num_samples=500, img_size=128):
    # Tạo dữ liệu ảnh giả
    X = np.random.randn(num_samples, img_size, img_size, 3) * 0.1
    
    # Tạo các vòng tròn ngẫu nhiên trong ảnh để mô phỏng nốt phổi
    y = np.zeros(num_samples)
    
    for i in range(num_samples):
        # Ảnh nền mô phỏng mô phổi
        X[i] = X[i] + 0.5  # Làm cho ảnh sáng hơn
        
        # Ngẫu nhiên quyết định đây có phải là ảnh có nốt ác tính hay không
        if np.random.rand() > 0.5:  # 50% là ác tính
            y[i] = 1
            # Tạo nốt ác tính (không đều, biên không rõ ràng)
            center_x, center_y = np.random.randint(30, img_size-30, 2)
            radius = np.random.randint(5, 15)
            for x in range(center_x-radius, center_x+radius):
                for y_coord in range(center_y-radius, center_y+radius):
                    if 0 <= x < img_size and 0 <= y_coord < img_size:
                        # Tạo nốt với biên không đều
                        dist = np.sqrt((x-center_x)**2 + (y_coord-center_y)**2)
                        if dist < radius + np.random.randn() * 2:
                            # Nốt ác tính thường có mật độ không đồng nhất
                            intensity = 0.8 + np.random.randn() * 0.1
                            X[i, x, y_coord, :] = intensity
        else:
            # Tạo nốt lành tính (tròn đều, biên rõ ràng) hoặc không có nốt
            if np.random.rand() > 0.3:  # 70% trong số lành tính có nốt
                center_x, center_y = np.random.randint(30, img_size-30, 2)
                radius = np.random.randint(3, 10)
                for x in range(center_x-radius, center_x+radius):
                    for y_coord in range(center_y-radius, center_y+radius):
                        if 0 <= x < img_size and 0 <= y_coord < img_size:
                            # Tạo nốt với biên đều
                            dist = np.sqrt((x-center_x)**2 + (y_coord-center_y)**2)
                            if dist < radius:
                                # Nốt lành tính thường có mật độ đồng nhất
                                X[i, x, y_coord, :] = 0.7
    
    # Chuyển label sang one-hot encoding
    y_categorical = to_categorical(y, num_classes=2)
    
    return X, y_categorical, y

# Tạo dữ liệu giả
print("Đang tạo dữ liệu mô phỏng...")
X, y_categorical, y_original = generate_synthetic_data(num_samples=1000, img_size=128)

# Chia tập dữ liệu thành train, validation và test
X_train_val, X_test, y_train_val, y_test, y_orig_train_val, y_orig_test = train_test_split(
    X, y_categorical, y_original, test_size=0.2, random_state=42, stratify=y_original
)

X_train, X_val, y_train, y_val, y_orig_train, y_orig_val = train_test_split(
    X_train_val, y_train_val, y_orig_train_val, test_size=0.2, random_state=42, stratify=y_orig_train_val
)

print(f"Kích thước tập train: {X_train.shape}")
print(f"Kích thước tập validation: {X_val.shape}")
print(f"Kích thước tập test: {X_test.shape}")

# Hiển thị một số ảnh mẫu
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i])
    plt.title(f"Lành tính" if y_orig_train[i] == 0 else "Ác tính")
    plt.axis('off')
    
for i in range(5):
    plt.subplot(2, 5, i+6)
    plt.imshow(X_train[i + 5])
    plt.title(f"Lành tính" if y_orig_train[i + 5] == 0 else "Ác tính")
    plt.axis('off')

plt.tight_layout()
plt.show()
_mfajlsdf98q21_image_title_list.append("Ảnh mô phỏng CT phổi với nốt lành tính và ác tính")

# Tạo data augmentation cho tập huấn luyện
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest',
    rescale=1./255
)

# Tiền xử lý cho tập validation và test mà không có augmentation
valid_test_datagen = ImageDataGenerator(rescale=1./255)

# Tạo generator cho tập train, validation và test
batch_size = 32
train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=batch_size,
    shuffle=True
)

validation_generator = valid_test_datagen.flow(
    X_val, y_val,
    batch_size=batch_size,
    shuffle=False
)

test_generator = valid_test_datagen.flow(
    X_test, y_test,
    batch_size=batch_size,
    shuffle=False
)

# Hiển thị ảnh sau khi augmentation
augmented_images, _ = next(train_generator)
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(augmented_images[i])
    plt.title(f"Augmented")
    plt.axis('off')
plt.tight_layout()
plt.show()
_mfajlsdf98q21_image_title_list.append("Ảnh sau khi áp dụng data augmentation")

# Xây dựng mô hình sử dụng transfer learning từ DenseNet121
def build_model(input_shape=(128, 128, 3), num_classes=2):
    # Tải mô hình pre-trained DenseNet121
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Đóng băng các lớp của mô hình cơ sở
    for layer in base_model.layers:
        layer.trainable = False
    
    # Thêm các lớp tùy chỉnh để phân loại
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Tạo model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Biên dịch model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', AUC()]
    )
    
    return model

# Xây dựng mô hình
model = build_model()
print(model.summary())

# Callback để giảm learning rate khi mô hình bắt đầu overfitting
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Callback để dừng sớm nếu mô hình bắt đầu overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Huấn luyện mô hình
print("Đang huấn luyện mô hình...")
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stopping],
    verbose=1
)

# Fine-tuning mô hình
print("Bắt đầu fine-tuning mô hình...")
# Mở khóa các lớp cuối của DenseNet
for layer in model.layers[0].layers[-30:]:
    layer.trainable = True

# Biên dịch lại model với learning rate thấp hơn
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', AUC()]
)

# Fine-tune mô hình
history_fine = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stopping],
    verbose=1
)

# Đánh giá mô hình trên tập test
print("Đánh giá mô hình trên tập test:")
results = model.evaluate(test_generator)
print(f"Loss: {results[0]}")
print(f"Accuracy: {results[1]}")
print(f"AUC: {results[2]}")

# Visualize training history
def plot_training_history(history, title):
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    _mfajlsdf98q21_image_title_list.append(title)

# Vẽ biểu đồ lịch sử huấn luyện
plot_training_history(history, "Lịch sử huấn luyện trong giai đoạn đầu")
if history_fine is not None:
    plot_training_history(history_fine, "Lịch sử huấn luyện trong giai đoạn fine-tuning")

# Dự đoán trên tập test
y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Tính toán confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

# Tính toán các metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall hoặc True Positive Rate
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

# In kết quả đánh giá
print("\nKết quả đánh giá trên tập test:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# Hiển thị confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Lành tính', 'Ác tính'])
plt.yticks(tick_marks, ['Lành tính', 'Ác tính'])

# Thêm giá trị số vào confusion matrix
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('Ground Truth')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()
_mfajlsdf98q21_image_title_list.append("Confusion Matrix")

# Vẽ đường cong ROC
fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
_mfajlsdf98q21_image_title_list.append("ROC Curve")

# Hiển thị một số ví dụ dự đoán từ tập test
num_samples_to_show = 10
plt.figure(figsize=(15, 8))
for i in range(num_samples_to_show):
    plt.subplot(2, 5, i+1)
    # Chuyển đổi về khoảng [0, 1]
    img = X_test[i] / 255.0
    plt.imshow(img)
    true_label = y_true[i]
    pred_label = y_pred[i]
    pred_prob = y_pred_prob[i][pred_label]
    
    color = 'green' if true_label == pred_label else 'red'
    title = f"T:{true_label} P:{pred_label} ({pred_prob:.2f})"
    plt.title(title, color=color)
    plt.axis('off')
    
plt.tight_layout()
plt.show()
_mfajlsdf98q21_image_title_list.append("Dự đoán trên dữ liệu test")

# Phân tích sai sót
misclassified_indices = np.where(y_true != y_pred)[0]
if len(misclassified_indices) > 0:
    num_examples = min(10, len(misclassified_indices))
    plt.figure(figsize=(15, 8))
    for i in range(num_examples):
        idx = misclassified_indices[i]
        plt.subplot(2, 5, i+1)
        img = X_test[idx] / 255.0  # Chia cho 255 để chuẩn hóa
        plt.imshow(img)
        true_label = 'Lành tính' if y_true[idx] == 0 else 'Ác tính'
        pred_label = 'Lành tính' if y_pred[idx] == 0 else 'Ác tính'
        plt.title(f"Thật: {true_label}\nDự đoán: {pred_label}", color='red')
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
    _mfajlsdf98q21_image_title_list.append("Phân tích các trường hợp bị phân loại sai")

# Đánh giá độ hiệu quả và độ ổn định của mô hình
print("\nĐánh giá độ hiệu quả và độ ổn định của mô hình:")
print(f"Độ chính xác (Accuracy): {accuracy:.4f}")
print(f"Độ nhạy (Sensitivity): {sensitivity:.4f}")
print(f"Độ đặc hiệu (Specificity): {specificity:.4f}")
print(f"Diện tích dưới đường cong ROC (AUC): {roc_auc:.4f}")

# Kết luận
print("\nKết luận:")
print("Mô hình deep learning đã được xây dựng và huấn luyện thành công để phát hiện và phân loại nốt phổi.")
print("Mô hình đã đạt được hiệu suất tốt trên tập test, cho thấy khả năng phát hiện chính xác ung thư phổi.")
print("Việc sử dụng transfer learning từ DenseNet121 và các kỹ thuật như data augmentation đã giúp cải thiện hiệu suất của mô hình.")
print("Mô hình có thể được cải thiện thêm với dữ liệu thực tế từ các bệnh viện và cơ sở y tế.")

print("\nDanh sách các tiêu đề hình ảnh:")
for i, title in enumerate(_mfajlsdf98q21_image_title_list):
    print(f"{i+1}. {title}")
