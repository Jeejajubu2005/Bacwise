# แก้ไขรหัสไฟล์ให้ถูกต้อง (คัดลอกเฉพาะ ID จากลิงก์)
SHAPE_MODEL_ID = '1BLR--79HAi8qgi0Z_dkW-diYkaPUGjRd'
COLOR_MODEL_ID = '1ZCOyWd4Py43b2gUqPowgKGN5LIxNsGfQ'

@st.cache_resource
def load_models():
    # ฟังก์ชันช่วยโหลดไฟล์จาก Drive
    def download_file(file_id, output):
        if not os.path.exists(output):
            # ระบบจะนำ ID ไปใส่หลัง ?id= เพื่อทำ Direct Download
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output, quiet=False)
    
    with st.spinner('กำลังโหลดโมเดลจาก Google Drive...'):
        download_file(SHAPE_MODEL_ID, 'shape_model.h5')
        download_file(COLOR_MODEL_ID, 'color_model.h5')
    
    shape_model = tf.keras.models.load_model('shape_model.h5')
    color_model = tf.keras.models.load_model('color_model.h5')
    return shape_model, color_model