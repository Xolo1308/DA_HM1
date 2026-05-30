"""
ĐỒ ÁN: PHÁT HIỆN WEBSITE PHISHING BẰNG MOBILENETV2
"""

import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# =========================
# 1. THAM SỐ CƠ BẢN
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(BASE_DIR, 'train')
test_dir = os.path.join(BASE_DIR, 'test')

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS_STAGE1 = 10      # Giai đoạn 1: Đông cứng backbone
EPOCHS_FINETUNE = 20    # Giai đoạn 2: Tinh chỉnh

MODEL_NAME = "MobileNetV2_Phishing_Detector"
MODEL_SAVE_PATH = os.path.join(BASE_DIR, f"{MODEL_NAME}.h5")

# =========================
# 2. KIỂM TRA DỮ LIỆU
# =========================
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Không tìm thấy {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Không tìm thấy {test_dir}")

print("="*60)
print("PHÁT HIỆN WEBSITE PHISHING - MOBILENETV2")
print("="*60)

# =========================
# 3. ĐỌC DỮ LIỆU
# =========================
print("\n[1/6] Đọc dữ liệu...")

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
print(f"Nhãn: {class_names} (0:{class_names[0]}, 1:{class_names[1]})")

# =========================
# 4. CÂN BẰNG DỮ LIỆU (CLASS WEIGHT)
# =========================
print("\n[2/6] Tính trọng số cho các lớp...")

train_path = pathlib.Path(train_dir)
so_luong = {}
for class_dir in sorted(train_path.iterdir()):
    if class_dir.is_dir():
        so_luong[class_dir.name] = len(list(class_dir.glob('*.*')))

tong_so = sum(so_luong.values())
trong_so = {i: tong_so/(len(class_names)*so_luong.get(name,1)) 
            for i, name in enumerate(class_names)}
print(f"Số lượng mỗi lớp: {so_luong}")
print(f"Trọng số: {trong_so}")

# =========================
# 5. TIỀN XỬ LÝ
# =========================
print("\n[3/6] Tiền xử lý ảnh...")

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)  # Chuẩn hóa về [-1, 1]
    return image, label

# Tăng cường dữ liệu
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

# =========================
# 6. XÂY DỰNG MÔ HÌNH
# =========================
print("\n[4/6] Xây dựng mô hình MobileNetV2...")

# Backbone MobileNetV2 (đã huấn luyện sẵn)
backbone = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
backbone.trainable = False  # Giai đoạn 1: đông cứng

# Phần đầu phân loại
inputs = layers.Input(shape=(224,224,3))
x = augmentation(inputs)
x = backbone(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(2, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =========================
# 7. HUẤN LUYỆN GIAI ĐOẠN 1
# =========================
print("\n[5/6] GIAI ĐOẠN 1: Huấn luyện phần đầu...")

history1 = model.fit(
    train_ds, epochs=EPOCHS_STAGE1, validation_data=val_ds,
    class_weight=trong_so,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(patience=2, factor=0.5)
    ]
)
print(f"Val accuracy sau giai đoạn 1: {max(history1.history['val_accuracy'])*100:.2f}%")

# =========================
# 8. FINE-TUNING GIAI ĐOẠN 2
# =========================
print("\n[5/6] GIAI ĐOẠN 2: Tinh chỉnh backbone...")

# Mở khóa 1 phần backbone để fine-tune
backbone.trainable = True
# Freeze batch normalization
for layer in backbone.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Compile lại với learning rate nhỏ hơn
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_ds, epochs=EPOCHS_FINETUNE, validation_data=val_ds,
    class_weight=trong_so,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.5)
    ]
)

# Lưu model
model.save(MODEL_SAVE_PATH)
print(f"Đã lưu model: {MODEL_SAVE_PATH}")

# =========================
# 9. ĐÁNH GIÁ TRÊN TEST
# =========================
print("\n[6/6] Đánh giá trên tập test...")

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Dự đoán chi tiết
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# Tính các chỉ số
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='macro')
rec = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

# Chỉ số từng lớp
prec_class = precision_score(y_true, y_pred, average=None)
rec_class = recall_score(y_true, y_pred, average=None)
f1_class = f1_score(y_true, y_pred, average=None)

print("\n" + "="*60)
print("KẾT QUẢ TỔNG HỢP")
print("="*60)
print(f"Accuracy : {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall   : {rec*100:.2f}%")
print(f"F1-Score : {f1*100:.2f}%")

print("\nKẾT QUẢ THEO TỪNG LỚP:")
for i, name in enumerate(class_names):
    print(f"  {name}: Precision={prec_class[i]*100:.2f}%, Recall={rec_class[i]*100:.2f}%, F1={f1_class[i]*100:.2f}%")

print("\n" + "="*60)
print("BÁO CÁO CHI TIẾT")
print("="*60)
print(classification_report(y_true, y_pred, target_names=class_names))

# =========================
# 10. VẼ BIỂU ĐỒ
# =========================
# Gộp lịch sử huấn luyện
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']
acc_hist = history1.history['accuracy'] + history2.history['accuracy']
val_acc_hist = history1.history['val_accuracy'] + history2.history['val_accuracy']

# Vẽ loss và accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss, 'b-', label='Train Loss')
plt.plot(val_loss, 'r-', label='Val Loss')
plt.axvline(x=len(history1.history['loss'])-0.5, color='gray', linestyle='--', label='Bắt đầu fine-tune')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Đường cong Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(acc_hist, 'b-', label='Train Acc')
plt.plot(val_acc_hist, 'r-', label='Val Acc')
plt.axvline(x=len(history1.history['accuracy'])-0.5, color='gray', linestyle='--', label='Bắt đầu fine-tune')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Đường cong Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, f"{MODEL_NAME}_curves.png"), dpi=150)
print(f"\nĐã lưu biểu đồ: {MODEL_NAME}_curves.png")

# Vẽ ma trận nhầm lẫn
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')

# Thêm số vào ô
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, f"{MODEL_NAME}_confusion.png"), dpi=150)
print(f"Đã lưu ma trận nhầm lẫn: {MODEL_NAME}_confusion.png")

# =========================
# 11. THÔNG TIN MÔ HÌNH
# =========================
total_params = model.count_params()
print("\n" + "="*60)
print("THÔNG TIN MÔ HÌNH")
print("="*60)
print(f"Tổng số tham số: {total_params:,} ({total_params/1e6:.2f} triệu)")
print(f"Số mẫu test: {len(y_true)}")

# =========================
# 12. KẾT LUẬN
# =========================
print("\n" + "="*60)
print("KẾT LUẬN")
print("="*60)
print(f"""
✅ MobileNetV2 đạt độ chính xác {acc*100:.2f}% trên tập test
✅ Phát hiện phishing - Recall: {rec_class[1]*100:.2f}%, Precision: {prec_class[1]*100:.2f}%
✅ Mô hình nhẹ ({total_params/1e6:.2f}M tham số), phù hợp triển khai thực tế
✅ Có thể tích hợp vào trình duyệt hoặc ứng dụng di động
""")

print("\n✨ HOÀN THÀNH! Các file đã tạo:")
print(f"   - Mô hình: {MODEL_SAVE_PATH}")
print(f"   - Biểu đồ: {MODEL_NAME}_curves.png")
print(f"   - Ma trận: {MODEL_NAME}_confusion.png")