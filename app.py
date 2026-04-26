import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# 1. ตั้งค่าหน้าเว็บ (ต้องอยู่บนๆ)
st.set_page_config(page_title="BacWise", page_icon="🔬")

# 2. รหัสไฟล์จาก Google Drive (ตรวจเช็ค ID ให้ดีนะครับ)
SHAPE_MODEL_ID = '1BLR--79HAi8qgi0Z_dkW-diYkaPUGjRd'
COLOR_MODEL_ID = '1ZCOyWd4Py43b2gUqPowgKGN5LIxNsGfQ'

# 3. ฟังก์ชันโหลดโมเดล
@st.cache_resource
def load_models():
    def download_file(file_id, output):
        if not os.path.exists(output):
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(url, output, quiet=False)
            except Exception as e:
                st.error(f"ดาวน์โหลดล้มเหลว: {e}")
    
    download_file(SHAPE_MODEL_ID, 'shape_model.h5')
    download_file(COLOR_MODEL_ID, 'color_model.h5')
    
    m1 = tf.keras.models.load_model('shape_model.h5')
    m2 = tf.keras.models.load_model('color_model.h5')
    return m1, m2

# --- เริ่มแสดงผลหน้าเว็บ ---
st.title("🔬 BacWise: Bacteria Analysis")
st.write("ระบบวิเคราะห์รูปร่างและสีแกรมของแบคทีเรียด้วย AI")

try:
    model_shape, model_color = load_models()
    st.success("เชื่อมต่อสมองกล AI เรียบร้อย!")
except Exception as e:
    st.error(f"ไม่สามารถโหลด AI ได้: {e}")

uploaded_file = st.file_uploader("เลือกรูปภาพแบคทีเรีย...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="รูปภาพที่อัปโหลด", use_container_width=True)
    
    with st.spinner('กำลังวิเคราะห์...'):
        # เตรียมรูป (224x224 ตามที่เทรนมา)
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ทำนายผล
        p1 = model_shape.predict(img_array)
        res_shape = "Cocci (ทรงกลม)" if p1[0][0] > 0.5 else "Bacilli (ทรงแท่ง)"
        
        p2 = model_color.predict(img_array)
        res_color = "Gram Positive (สีม่วง)" if p2[0][0] > 0.5 else "Gram Negative (สีชมพู)"

    # แสดงผลลัพธ์แบบสวยงาม
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("รูปร่าง", res_shape)
    c2.metric("ชนิดสี", res_color)
    st.info(f"สรุปผล: แบคทีเรียในภาพมีลักษณะเป็น {res_shape} และติดสีแบบ {res_color}")
