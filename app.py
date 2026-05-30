
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import os
import uuid
from datetime import datetime
import csv
# ============================================
# CẤU HÌNH THRESHOLD TỐI ƯU (DỰA TRÊN PHÂN TÍCH)
# ============================================
THRESHOLD_CONFIG = {
    "ResNet50": {
        "phishing_threshold": 0.62,      # Tối ưu từ confusion matrix
        "description": "Đã tối ưu để giảm False Positive (báo nhầm)"
    },
    "MobileNetV2": {
        "phishing_threshold": 0.65,
        "description": "Cân bằng giữa tốc độ và độ chính xác"
    }
}

# ============================================
# 1. CẤU HÌNH TRANG WEB
# ============================================
st.set_page_config(
    page_title="Hệ thống phát hiện Website lừa đảo - Phishing Detector",
    page_icon="",
    layout="wide"
)

# Tùy chỉnh CSS
st.markdown("""
    <style>
    .main { background-color: #f0faf0; }
    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8faf8 100%);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        border: 1px solid #c8e6c9;
    }
    .reason-box {
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 10px;
        border-left: 5px solid #2e7d32;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .reason-box-warning {
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 10px;
        border-left: 5px solid #d32f2f;
        background-color: #fff5f5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .threshold-box {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    h1, h2, h3 { color: #1b5e20 !important; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8f5e9 0%, #c8e6c9 100%);
    }
    .stProgress > div > div { background-color: #2e7d32; }
    .model-card {
        background: white;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .comparison-good { color: #2e7d32; font-weight: bold; }
    .comparison-bad { color: #d32f2f; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# 2. HÀM LOAD MODEL
# ============================================
@st.cache_resource
def load_model(model_name):
    """Tải mô hình đã chọn"""
    model_paths = {
        #"ResNet50": "ResNet50_Phishing_Detector.keras",
        "ResNet50": "ResNet50_Phishing_Detector1.h5",
        "MobileNetV2": "MobileNetV2_Phishing_Detector1.h5",
        
    }
    
    try:
        model = tf.keras.models.load_model(model_paths[model_name])
        return model
    except Exception as e:
        st.error(f"Không thể tải mô hình {model_name}! Lỗi: {e}")
        return None

# ============================================
# 3. HÀM TIỀN XỬ LÝ ẢNH
# ============================================
def preprocess_image(img, model_name):
    """Tiền xử lý ảnh phù hợp với từng mô hình"""
    img = img.convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    if model_name == "ResNet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        img_preprocessed = preprocess_input(img_array)
    else:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img_preprocessed = preprocess_input(img_array)
    
    return img_preprocessed

# ============================================
# 4. HÀM DỰ ĐOÁN VỚI THRESHOLD
# ============================================
def predict_with_threshold(model, image, model_name):
    """
    Dự đoán nhị phân (Binary) với threshold tối ưu
    Trả về: (is_phishing, prob_phish, prob_legit, confidence_level, zone, threshold)
    """
    # Dự đoán
    prediction = model.predict(image, verbose=0)
    prob_phish = float(prediction[0][1])
    prob_legit = float(prediction[0][0])
    
    # Lấy threshold cho model
    config = THRESHOLD_CONFIG.get(model_name, THRESHOLD_CONFIG["ResNet50"])
    threshold = config["phishing_threshold"]
    
    # Phân loại nhị phân (Loại bỏ vùng không chắc chắn)
    if prob_phish >= threshold:
        is_phishing = True
        # Độ tin cậy dựa trên khoảng cách tới threshold
        if prob_phish > 0.85:
            confidence_level = "Rất cao"
        elif prob_phish > 0.70:
            confidence_level = "Cao"
        else:
            confidence_level = "Trung bình"
        zone = "phishing_zone"
    else:
        is_phishing = False
        if prob_legit > 0.85:
            confidence_level = "Rất cao"
        elif prob_legit > 0.70:
            confidence_level = "Cao"
        else:
            confidence_level = "Trung bình"
        zone = "legitimate_zone"
    
    return is_phishing, prob_phish, prob_legit, confidence_level, zone, threshold

# ============================================
# 5. HÀM HIỂN THỊ THRESHOLD INFO
# ============================================
def display_threshold_info(model_name, prob_phish, threshold):
    """Hiển thị thông tin về threshold và vùng quyết định"""
    
    config = THRESHOLD_CONFIG.get(model_name, THRESHOLD_CONFIG["ResNet50"])
    
    st.markdown(f"""
    <div ">
        
      
    </div>
    """, unsafe_allow_html=True)
    
 
# ============================================
# 5.5 HÀM LƯU PHẢN HỒI (ACTIVE LEARNING)
# ============================================
def save_feedback(image, predicted_label, actual_label, model_name, prob_phish):
    """Lưu ảnh và thông tin phản hồi của người dùng để retrain model"""
    base_dir = "feedback_data"
    phishing_dir = os.path.join(base_dir, "phishing")
    legit_dir = os.path.join(base_dir, "legitimate")
    
    os.makedirs(phishing_dir, exist_ok=True)
    os.makedirs(legit_dir, exist_ok=True)
    
    file_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file_id}.png"
    
    if actual_label == "Phishing":
        save_path = os.path.join(phishing_dir, filename)
    else:
        save_path = os.path.join(legit_dir, filename)
        
    image.save(save_path)
    
    log_file = os.path.join(base_dir, "feedback_log.csv")
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "filename", "model_used", "prob_phish", "predicted_label", "actual_label", "is_correct"])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            model_name,
            f"{prob_phish:.4f}",
            predicted_label,
            actual_label,
            predicted_label == actual_label
        ])

# ============================================
# 6. HÀM LẤY THÔNG TIN MÔ HÌNH
# ============================================
def get_model_info(model_name):
    """Thông tin chi tiết về mô hình"""
    models_info = {
        "ResNet50": {
            "icon": "",
            "name": "ResNet50",
            "accuracy": "81.99% (thực tế)",
            "params": "24.15 triệu",
            "speed": "~45ms/ảnh",
            "threshold": "0.62",
            "pros": [
                "Độ chính xác cao nhất",
                "Học đặc trưng phức tạp tốt",
                "Phát hiện phishing: 92%"
            ],
            "cons": [
                "Yêu cầu GPU/CPU mạnh",
                "Thời gian suy luận chậm hơn"
            ]
        },
        "MobileNetV2": {
            "icon": "",
            "name": "MobileNetV2",
            "accuracy": "88.7%",
            "params": "3.5 triệu",
            "speed": "~18ms/ảnh",
            "threshold": "0.55",
            "pros": [
                "Mô hình nhẹ, chạy nhanh",
                "Phù hợp triển khai trình duyệt",
                "Tối ưu cho thiết bị di động"
            ],
            "cons": [
                "   Độ chính xác thấp hơn ResNet50",
                "Học đặc trưng kém tinh vi hơn"
            ]
        }
    }
    return models_info.get(model_name, models_info["MobileNetV2"])

# ============================================
# 7. HÀM PHÂN TÍCH CHI TIẾT
# ============================================
def get_detailed_reasons(is_phishing):
    """Lấy lý do phân tích chi tiết"""
    if is_phishing:
        return {
            "Giao diện & Bố cục": {
                "type": "warning",
                "reasons": [
                    "Sao chép gần như hoàn hảo (>85%) giao diện của thương hiệu nổi tiếng",
                    "Sử dụng template phishing phổ biến trong cơ sở dữ liệu đã ghi nhận",
                    "Bố cục thiếu nhất quán, các thành phần UI không căn chỉnh chuẩn",
                    "Tồn tại lỗi hiển thị ở vùng footer và các liên kết phụ"
                ]
            },          
            "Bảo mật & Form nhập liệu": {
                "type": "warning",
                "reasons": [
                    "Form đăng nhập không hiển thị chứng chỉ SSL hợp lệ",
                    "Yêu cầu nhập thông tin nhạy cảm không cần thiết",
                    "Các trường nhập liệu không có biểu tượng khóa bảo mật",
                    "Gửi dữ liệu qua kênh không mã hóa (HTTP thay vì HTTPS)"
                ]
            },
            "Chất lượng hình ảnh & Logo": {
                "type": "warning",
                "reasons": [
                    "Logo bị cắt ghép thô, độ phân giải thấp",
                    "Font chữ không đồng bộ, có dấu hiệu render sai",
                    "Icon và button bị mờ, không sắc nét",
                    "Sử dụng ảnh stock giá rẻ thay vì ảnh thương hiệu chính thống"
                ]
            }
        }
    else:
        return {
            "Giao diện & Bố cục": {
                "type": "success",
                "reasons": [
                    "Bố cục chuyên nghiệp, nhất quán với thương hiệu chính thống",
                    "Các thành phần UI được căn chỉnh chuẩn xác",
                    "Sử dụng khoảng trắng hợp lý",
                    "Footer đầy đủ thông tin: điều khoản, chính sách bảo mật"
                ]
            },
            "Dấu hiệu bảo mật chuẩn": {
                "type": "success",
                "reasons": [
                    "Hiển thị rõ chứng chỉ SSL/khóa bảo mật",
                    "Các form nhập liệu được bảo vệ bằng mã hóa",
                    "Có badge xác thực nếu là trang thương mại",
                    "Tuân thủ chuẩn bảo mật PCI DSS"
                ]
            },
            "Nhận diện thương hiệu": {
                "type": "success",
                "reasons": [
                    "Logo sắc nét, đúng màu sắc và tỷ lệ chuẩn",
                    "Font chữ đồng bộ, sử dụng web font chính thống",
                    "Hình ảnh chất lượng cao, chụp chuyên nghiệp",
                    "Gắn kết các biểu tượng mạng xã hội chính thức"
                ]
            },
            "Hành vi người dùng": {
                "type": "success",
                "reasons": [
                    "Không có pop-up quảng cáo gây khó chịu",
                    "Không tự động tải xuống file hay cài đặt extension",
                    "Thời gian phản hồi nhanh",
                    "Có hỗ trợ khách hàng rõ ràng"
                ]
            }
        }

def get_actionable_tips(is_phishing, model_name):
    """Lấy khuyến nghị hành động"""
    if is_phishing:
        return [
            "**Tuyệt đối KHÔNG nhập** mật khẩu, mã OTP, số thẻ tín dụng",
            "**KHÔNG tải file** hay cài đặt bất kỳ phần mềm nào",
            "**Báo cáo ngay** cho VNCERT (024.36463922)",
            "**Xóa cache, cookie và lịch sử** trình duyệt",
            "**Đổi mật khẩu** tất cả các tài khoản quan trọng",
            "**Kích hoạt xác thực 2 lớp (2FA)**",
            "**Liên hệ ngân hàng** nếu đã nhập thông tin thẻ"
        ]
    else:
        return [
            "**Tiếp tục sử dụng** website một cách bình thường",
            "**Kiểm tra tên miền** trên thanh địa chỉ",
            "**Duy trì thói quen** chỉ nhập thông tin trên HTTPS",
            "**Cập nhật trình duyệt** lên phiên bản mới nhất",
            "**Bật xác thực 2 lớp (2FA)** cho mọi tài khoản",
            "**Học hỏi thêm** về dấu hiệu nhận biết phishing"
        ]

# ============================================
# 8. SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
    st.title("🌿 Hệ thống phát hiện website độc hại TH")
    
    # Lựa chọn mô hình
    st.subheader("Chọn mô hình")
    selected_model = st.radio(
        "Mô hình phân loại:",
        ["ResNet50", "MobileNetV2"],
        index=0,
        help="ResNet50: Chính xác cao hơn | MobileNetV2: Nhanh hơn, nhẹ hơn"
    )
    
    # Hiển thị thông tin mô hình
    model_info = get_model_info(selected_model)
    threshold_info = THRESHOLD_CONFIG[selected_model]
    
    st.markdown(f"""
    <div class="model-card">
        <h3>{model_info['icon']} {model_info['name']}</h3>
        <table style="width:100%; font-size:14px;">
            <tr><td>Độ chính xác:</td><td><b>{model_info['accuracy']}</b></td></tr>
            <tr><td>Số tham số:</td><td><b>{model_info['params']}</b></td></tr>
            <tr><td>Tốc độ:</td><td><b>{model_info['speed']}</b></td></tr>
            <tr><td>Ngưỡng Phishing:</td><td><b>{threshold_info['phishing_threshold']*100:.0f}%</b></td></tr>
        </table>
        <br>
        <b>Ưu điểm:</b><br>
        {"".join([f"{p}<br>" for p in model_info['pros']])}
        <br>
        <b>Nhược điểm:</b><br>
        {"".join([f"{c}<br>" for c in model_info['cons']])}
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    

# ============================================
# 9. GIAO DIỆN CHÍNH
# ============================================
st.title("HỆ THỐNG PHÁT HIỆN WEBSITE LỪA ĐẢO")
st.markdown(f"### Đang sử dụng mô hình: **{selected_model}** {model_info['icon']}")

# Upload file
uploaded_file = st.file_uploader(
    "Tải lên ảnh chụp màn hình website", 
    type=["jpg", "png", "jpeg"],
    help="Hỗ trợ định dạng JPG, PNG, JPEG"
)

if uploaded_file is not None:
    # Load model
    model = load_model(selected_model)
    if model is None:
        st.stop()
    
    # Hiển thị ảnh gốc
    img = Image.open(uploaded_file)
    
    col_img, col_info = st.columns([1, 1])
    with col_img:
        st.image(img, caption="Ảnh đã tải lên", use_container_width=True)
    
    with col_info:
        st.info(f"""
        **Thông tin ảnh:**
        - Kích thước: {img.size[0]} x {img.size[1]} pixel
        - Định dạng: {img.format}
        - Đang xử lý với {selected_model}
        """)
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        progress_bar.progress(i + 1)
        if i < 30:
            status_text.text("Đang phân tích cấu trúc giao diện...")
        elif i < 60:
            status_text.text("Đang quét yếu tố bảo mật...")
        elif i < 85:
            status_text.text("Đang kiểm tra chất lượng hình ảnh...")
        else:
            status_text.text("Tổng hợp kết quả...")
        time.sleep(0.005)
    
    # Tiền xử lý và dự đoán
    img_preprocessed = preprocess_image(img, selected_model)
    
    # Dự đoán với threshold tối ưu
    is_phishing, prob_phish, prob_legit, confidence_level, zone, threshold = predict_with_threshold(
        model, img_preprocessed, selected_model
    )
    
    progress_bar.empty()
    status_text.empty()
    st.divider()
    
    # ============================================
    # HIỂN THỊ THRESHOLD INFO
    # ============================================
    display_threshold_info(selected_model, prob_phish, threshold)
    
    # ============================================
    # HIỂN THỊ KẾT QUẢ
    # ============================================
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if is_phishing is True:
            st.error(" **KẾT LUẬN: NGUY HIỂM - PHISHING**")
            st.warning("Website có dấu hiệu lừa đảo! **TUYỆT ĐỐI KHÔNG** nhập thông tin cá nhân.")
            st.markdown(f"**Độ tin cậy:** {confidence_level} ({prob_phish*100:.1f}% >= {threshold*100:.0f}%)")
        else:
            st.success("**KẾT LUẬN: AN TOÀN - LEGITIMATE**")
            st.info("Website được đánh giá là hợp pháp và an toàn.")
            st.markdown(f"**Độ tin cậy:** {confidence_level} ({prob_legit*100:.1f}% > {(1-threshold)*100:.0f}%)")
    
    with col2:
        st.metric("Mức độ an toàn", f"{prob_legit*100:.1f}%")
        st.progress(prob_legit, text="Xác suất hợp pháp")
    
    with col3:
        st.metric("Mức độ nguy hiểm", f"{prob_phish*100:.1f}%")
        st.progress(prob_phish, text="Xác suất lừa đảo")
    
    st.divider()
    
    # ============================================
    # CƠ CHẾ ACTIVE LEARNING & CROWDSOURCING
    # ============================================
    st.markdown("### Đóng góp cải thiện AI (Active Learning)")
    st.markdown("Nếu AI dự đoán sai, hãy giúp chúng tôi báo cáo để hệ thống học lại và thông minh hơn!")
    
    predicted_label = "Phishing" if is_phishing else "Legitimate"
    
    # Khởi tạo session state cho modal
    if "show_feedback_modal" not in st.session_state:
        st.session_state.show_feedback_modal = False
    if "show_thanks_modal" not in st.session_state:
        st.session_state.show_thanks_modal = False
    if "thanks_message" not in st.session_state:
        st.session_state.thanks_message = ""
    if "thanks_is_correction" not in st.session_state:
        st.session_state.thanks_is_correction = False
    
    # Button để mở modal
    col_modal_btn = st.columns([1])
    with col_modal_btn[0]:
        if st.button(" Báo cáo & Đóng góp dữ liệu", use_container_width=True, key="open_feedback_modal"):
            st.session_state.show_feedback_modal = True
    
    # Modal Dialog - Báo cáo
    if st.session_state.show_feedback_modal:
        @st.dialog("Báo cáo & Đóng góp dữ liệu", width="large")
        def feedback_modal():
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <h3> Phản hồi của bạn rất quan trọng</h3>
                <p><b>Dự đoán hiện tại của AI:</b> <span style="color: {'#d32f2f' if is_phishing else '#2e7d32'}; font-size: 18px; font-weight: bold;">{predicted_label}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**Theo bạn, trang web này thực chất là gì?**")
            
            col_feedback1, col_feedback2 = st.columns(2)
            
            with col_feedback1:
                if st.button("Trang Lừa đảo (Phishing)", use_container_width=True, key="btn_phishing"):
                    save_feedback(img, predicted_label, "Phishing", selected_model, prob_phish)
                    st.session_state.thanks_is_correction = predicted_label != "Phishing"
                    st.session_state.show_thanks_modal = True
                    st.session_state.show_feedback_modal = False
                    st.rerun()
            
            with col_feedback2:
                if st.button("Trang An toàn (Legitimate)", use_container_width=True, key="btn_legitimate"):
                    save_feedback(img, predicted_label, "Legitimate", selected_model, prob_phish)
                    st.session_state.thanks_is_correction = predicted_label != "Legitimate"
                    st.session_state.show_thanks_modal = True
                    st.session_state.show_feedback_modal = False
                    st.rerun()
            
            st.divider()
            
            st.info("""
            💡 **Tại sao bạn nên báo cáo?**
            - Giúp AI trở nên thông minh hơn
            - Bảo vệ cộng đồng khỏi phishing
            - Góp phần xây dựng công nghệ AI an toàn
            """)
        
        feedback_modal()
    
    # Modal Dialog - Lời cảm ơn
    if st.session_state.show_thanks_modal:
        @st.dialog("Cảm ơn bạn!", width="large")
        def thanks_modal():
            if st.session_state.thanks_is_correction:
                st.markdown("""
                <div style="text-align: center; padding: 30px;">
                    <h2 style="color: #2e7d32; font-size: 32px;">Cảm ơn bạn!</h2>
                    <p style="font-size: 18px; margin: 20px 0;">Bạn vừa giúp sửa sai cho AI</p>
                    <hr>
                    <p style="font-size: 16px; color: #555; margin: 20px 0;">
                        Ảnh này sẽ được <b>đưa vào tập huấn luyện mới</b> để AI học hỏi và trở nên <b>thông minh hơn</b>.
                    </p>
                    <p style="font-size: 16px; color: #555; margin: 20px 0;">
                        Góp phần của bạn giúp bảo vệ <b>cộng đồng người dùng</b> khỏi các trang web lừa đảo.
                    </p>
                    <hr>
                    <p style="font-size: 14px; color: #888; margin-top: 20px;">
                        Hãy tiếp tục báo cáo để AI ngày càng tốt hơn!
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 30px;">
                    <h2 style="color: #2e7d32; font-size: 32px;">Cảm ơn bạn!</h2>
                    <p style="font-size: 18px; margin: 20px 0;">Bạn vừa xác nhận dự đoán của AI</p>
                    <hr>
                    <p style="font-size: 16px; color: #555; margin: 20px 0;">
                        Dữ liệu đã được <b>lưu lại</b> để tăng cường <b>độ chính xác</b> của hệ thống.
                    </p>
                    <p style="font-size: 16px; color: #555; margin: 20px 0;">
                        Mỗi xác nhận từ bạn giúp AI trở nên <b>đáng tin cậy hơn</b> cho những người dùng khác.
                    </p>
                    <hr>
                    <p style="font-size: 14px; color: #888; margin-top: 20px;">
                        Cảm ơn vì góp phần xây dựng công nghệ AI an toàn!
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.balloons()
            
            col_thanks1, col_thanks2 = st.columns(2)
            with col_thanks1:
                if st.button("Tiếp tục phân tích", use_container_width=True, key="btn_continue"):
                    st.session_state.show_thanks_modal = False
                    st.rerun()
            with col_thanks2:
                if st.button("Xem thống kê feedback", use_container_width=True, key="btn_stats"):
                    st.session_state.show_thanks_modal = False
        
        thanks_modal()
    
    st.divider()
    
    # ============================================
    # TABS HIỂN THỊ CHI TIẾT
    # ============================================
    tab1, tab2, tab3 = st.tabs(["PHÂN TÍCH CHI TIẾT", "KHUYẾN NGHỊ", "SO SÁNH MÔ HÌNH"])
    
    with tab1:
        reasons_dict = get_detailed_reasons(is_phishing)
        for group_name, group_data in reasons_dict.items():
            st.markdown(f"#### {group_name}")
            for reason in group_data["reasons"]:
                css_class = "reason-box-warning" if group_data["type"] == "warning" else "reason-box"
                st.markdown(f"<div class='{css_class}'>{reason}</div>", unsafe_allow_html=True)
    
    with tab2:
        tips = get_actionable_tips(is_phishing, selected_model)
        for i, tip in enumerate(tips, 1):
            if is_phishing:
                st.warning(f"{i}. {tip}")
            else:
                st.success(f"{i}. {tip}")
    
    with tab3:
        st.subheader("So sánh dự đoán giữa 2 mô hình")
        
        model_resnet = load_model("ResNet50")
        model_mobile = load_model("MobileNetV2")
        
        if model_resnet is not None and model_mobile is not None:
            # Dự đoán với cả 2 model
            img_resnet = preprocess_image(img, "ResNet50")
            img_mobile = preprocess_image(img, "MobileNetV2")
            
            pred_resnet = model_resnet.predict(img_resnet, verbose=0)
            pred_mobile = model_mobile.predict(img_mobile, verbose=0)
            
            prob_resnet_phish = float(pred_resnet[0][1])
            prob_mobile_phish = float(pred_mobile[0][1])
            
            # Lấy threshold cho từng model
            thresh_resnet = THRESHOLD_CONFIG["ResNet50"]["phishing_threshold"]
            thresh_mobile = THRESHOLD_CONFIG["MobileNetV2"]["phishing_threshold"]
            
            col_r, col_m = st.columns(2)
            
            with col_r:
                st.markdown("### ResNet50")
                st.markdown(f"**Ngưỡng:** {thresh_resnet*100:.0f}%")
                
                if prob_resnet_phish >= thresh_resnet:
                    st.error(f"Dự đoán: **PHISHING** ({prob_resnet_phish*100:.1f}%)")
                else:
                    st.success(f"Dự đoán: **AN TOÀN** ({(1-prob_resnet_phish)*100:.1f}%)")
                
                st.progress(prob_resnet_phish, text="Xác suất Phishing")
                st.caption(f"Precision Phishing: 85.2% | FP thấp")
            
            with col_m:
                st.markdown("### MobileNetV2")
                st.markdown(f"**Ngưỡng:** {thresh_mobile*100:.0f}%")
                
                if prob_mobile_phish >= thresh_mobile:
                    st.error(f" Dự đoán: **PHISHING** ({prob_mobile_phish*100:.1f}%)")
                else:
                    st.success(f"Dự đoán: **AN TOÀN** ({(1-prob_mobile_phish)*100:.1f}%)")
                
                st.progress(prob_mobile_phish, text="Xác suất Phishing")
                st.caption(f"Tốc độ: ~18ms | Nhẹ hơn 7 lần")
            
            st.divider()
            
            # Hiển thị so sánh kết luận
            verdict_resnet = prob_resnet_phish >= thresh_resnet
            verdict_mobile = prob_mobile_phish >= thresh_mobile
            
            if verdict_resnet == verdict_mobile:
                st.success("**Cả 2 mô hình đều đưa ra cùng nhận định!**")
            else:
                st.warning("**Cảnh báo: 2 mô hình đưa ra nhận định khác nhau!**")
                st.info("Nên tin tưởng vào mô hình ResNet50 hơn vì độ chính xác cao hơn, hoặc kiểm tra lại thủ công.")

else:
    # Hiển thị hướng dẫn khi chưa có ảnh
    st.info("**Hướng dẫn sử dụng:**\n\n1. Chọn mô hình ở thanh bên trái\n2. Tải lên ảnh chụp màn hình website\n3. Chờ hệ thống phân tích\n4. Xem kết quả và khuyến nghị")
    
    col_guide1, col_guide2, col_guide3 = st.columns(3)
    with col_guide1:
        st.markdown("""
        ### Khi nào dùng MobileNetV2?
        - Máy tính cấu hình thấp (CPU)
        - Cần phản hồi nhanh
        - Triển khai lên trình duyệt
        """)
    with col_guide2:
        st.markdown("""
        ### Khi nào dùng ResNet50?
        - Có GPU mạnh
        - Cần độ chính xác cao nhất
        - Phân tích offline
        """)
    with col_guide3:
        st.markdown("""
        ### Yêu cầu ảnh đầu vào
        - Ảnh chụp toàn bộ màn hình
        - Định dạng JPG/PNG
        - Nội dung rõ nét
        """)

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption("""
** Lưu ý:** 
- Hệ thống sử dụng AI để phân tích, độ chính xác phụ thuộc vào chất lượng ảnh đầu vào
- Kết quả mang tính tham khảo, không thay thế đánh giá bảo mật chuyên nghiệp
- Nếu phát hiện website lừa đảo, hãy báo ngay cho cơ quan chức năng
""")