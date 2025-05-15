import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# تحميل الموديل
model = load_model("my_model.h5")

# شكل الإدخال المتوقع
input_shape = model.input_shape[1:3]  # أمثلة: (72, 72)
st.write("🔍 شكل الإدخال المتوقع من النموذج:", input_shape)

# إعدادات الصفحة
st.title("تصنيف أوراق الفراولة 🍓")
st.write("حمّل صورة ورقة فراولة لمعرفة إذا كانت صحية أم لا.")

# تحميل الصورة
uploaded_file = st.file_uploader("اختار صورة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="الصورة المُحمّلة", use_column_width=True)

    # تجهيز الصورة للتوقع
    img = img.resize(input_shape)  # استخدام الحجم المناسب للنموذج
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # التوقع
    prediction = model.predict(img_array)[0][0]

    # عرض النتيجة
    st.write("## النتيجة:")
    if prediction < 0.5:
        st.success(f"✅ {(1-prediction)*100} الورقة صحية" )
    else:
        st.error(f"❌ {1-(prediction)*100} الورقة غير صحية")

