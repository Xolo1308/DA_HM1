"""
ĐỒ ÁN: PHÁT HIỆN WEBSITE PHISHING BẰNG CNN - ResNet50
"""

import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# =========================
# 1. THAM SỐ
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(BASE_DIR, 'train')
test_dir = os.path.join(BASE_DIR, 'test')

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS_STAGE1 = 10
EPOCHS_FINETUNE = 30

MODEL_NAME = "ResNet50_Phishing_Detector"
MODEL_SAVE_PATH = os.path.join(BASE_DIR, f"{MODEL_NAME}.h5")

# =========================
# 2. KIỂM TRA DỮ LIỆU
# =========================
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Không tìm thấy {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Không tìm thấy {test_dir}")

print("="*60)
print("PHÁT HIỆN WEBSITE PHISHING - ResNet50")
print("="*60)

# =========================
# 3. ĐỌC DỮ LIỆU
# =========================
print("\n[1/6] Đang đọc dữ liệu...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, validation_split=0.2, subset='training', seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, validation_split=0.2, subset='validation', seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle=False
)

class_names = train_ds.class_names
print(f"Classes: {class_names}")

# =========================
# 4. TIỀN XỬ LÝ
# =========================
print("\n[2/6] Tiền xử lý dữ liệu...")

def preprocess(image, label):
    return preprocess_input(tf.cast(image, tf.float32)), label

train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

# Data augmentation đơn giản
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
])

# Class weights
import pathlib
class_counts = {}
for class_dir in sorted(pathlib.Path(train_dir).iterdir()):
    if class_dir.is_dir():
        class_counts[class_dir.name] = len(list(class_dir.glob('*.*')))

total = sum(class_counts.values())
class_weight = {i: total/(len(class_names)*class_counts.get(n,1)) 
                for i, n in enumerate(class_names)}
print(f"Class weights: {class_weight}")

# =========================
# 5. XÂY DỰNG MÔ HÌNH
# =========================
print("\n[3/6] Xây dựng mô hình ResNet50...")

base_model = ResNet50(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = layers.Input(shape=(224,224,3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =========================
# 6. HUẤN LUYỆN
# =========================
print("\n[4/6] Giai đoạn 1 - Frozen backbone...")
history1 = model.fit(train_ds, epochs=EPOCHS_STAGE1, validation_data=val_ds,
    class_weight=class_weight, 
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True),ReduceLROnPlateau(patience=2, factor=0.5)])

print("\n[5/6] Giai đoạn 2 - Fine-tuning...")
base_model.trainable = True
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), 
              loss='categorical_crossentropy', metrics=['accuracy'])

history2 = model.fit(train_ds, epochs=EPOCHS_FINETUNE, validation_data=val_ds,
        class_weight=class_weight,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True),
                               ReduceLROnPlateau(patience=3, factor=0.5)])

# Lưu model
model.save(MODEL_SAVE_PATH)
print(f"Đã lưu model: {MODEL_SAVE_PATH}")

# =========================
# 7. ĐÁNH GIÁ
# =========================
print("\n[6/6] Đánh giá trên test set...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Dự đoán
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# Chỉ số
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='macro')
rec = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print("\n" + "="*60)
print("KẾT QUẢ")
print("="*60)
print(f"Accuracy : {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall   : {rec*100:.2f}%")
print(f"F1-Score : {f1*100:.2f}%")

print("\nChi tiết từng lớp:")
print(classification_report(y_true, y_pred, target_names=class_names))

# =========================
# 8. VẼ BIỂU ĐỒ
# =========================
# Gộp lịch sử
all_loss = history1.history['loss'] + history2.history['loss']
all_val_loss = history1.history['val_loss'] + history2.history['val_loss']
all_acc = history1.history['accuracy'] + history2.history['accuracy']
all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']

# Vẽ loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(all_loss, 'b-', label='Train Loss')
plt.plot(all_val_loss, 'r-', label='Val Loss')
plt.axvline(x=len(history1.history['loss'])-0.5, color='gray', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# Vẽ accuracy
plt.subplot(1,2,2)
plt.plot(all_acc, 'b-', label='Train Acc')
plt.plot(all_val_acc, 'r-', label='Val Acc')
plt.axvline(x=len(history1.history['accuracy'])-0.5, color='gray', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, f"{MODEL_NAME}_curves.png"), dpi=150)
print(f"\nĐã lưu biểu đồ: {MODEL_NAME}_curves.png")

# Vẽ ma trận nhầm lẫn
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, f"{MODEL_NAME}_confusion.png"), dpi=150)
print(f"Đã lưu ma trận: {MODEL_NAME}_confusion.png")

# =========================
# 9. KẾT LUẬN
# =========================
print("\n" + "="*60)
print("KẾT LUẬN")
print("="*60)
print(f"""
Mô hình ResNet50 đạt độ chính xác {acc*100:.2f}% trên tập test
Phát hiện phishing với F1-score: {f1*100:.2f}%
Có thể ứng dụng cảnh báo web lừa đảo thời gian thực
""")

print("\n✨ HOÀN THÀNH!")