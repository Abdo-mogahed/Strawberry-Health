import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import io
from inference_sdk import InferenceHTTPClient

# القاموس الخاص بالأمراض
plant_diseases = {
    "Angular Leafspot": {
        "Cause": "Bacterial infection (Pseudomonas syringae pv. lachrymans)",
        "Treatment": "Use copper-based fungicides and remove infected leaves",
        "Prevention": "Avoid overhead irrigation, use disease-resistant varieties, rotate crops"
    },
    "Anthracnose Fruit Rot": {
        "Cause": "Fungal infection (Colletotrichum spp.)",
        "Treatment": "Apply fungicides like chlorothalonil or copper sprays, remove infected fruits",
        "Prevention": "Ensure good air circulation, avoid fruit injury, sanitize tools"
    },
    "Blossom Blight": {
        "Cause": "Fungal infection (e.g., Botrytis cinerea or Monilinia spp.)",
        "Treatment": "Use fungicides such as iprodione or thiophanate-methyl",
        "Prevention": "Avoid high humidity, prune for air circulation, remove infected blossoms"
    },
    "Gray Mold": {
        "Cause": "Fungal infection (Botrytis cinerea)",
        "Treatment": "Apply fungicides like fenhexamid, remove and destroy infected parts",
        "Prevention": "Improve ventilation, avoid excess moisture, remove plant debris"
    },
    "Leaf Spot": {
        "Cause": "Fungal or bacterial pathogens (e.g., Septoria, Cercospora)",
        "Treatment": "Use appropriate fungicides or bactericides depending on the cause",
        "Prevention": "Avoid wetting foliage, use resistant varieties, clean up fallen leaves"
    },
    "Powdery Mildew Fruit": {
        "Cause": "Fungal infection (e.g., Podosphaera spp.)",
        "Treatment": "Spray with sulfur-based or systemic fungicides like myclobutanil",
        "Prevention": "Ensure good air circulation, avoid over-fertilizing, use resistant cultivars"
    },
    "Powdery Mildew Leaf": {
        "Cause": "Fungal infection (e.g., Erysiphe spp.)",
        "Treatment": "Apply fungicides like neem oil, potassium bicarbonate, or sulfur",
        "Prevention": "Prune overcrowded plants, reduce humidity, plant in sunny areas"
    }
}

# تحميل الموديل
model = load_model("my_model.h5")

# إعدادات الصفحة
st.title("تصنيف أوراق الفراولة 🍓")
st.write("حمّل صورة ورقة فراولة لمعرفة إذا كانت صحية أم لا.")

# شكل الإدخال المتوقع
input_shape = model.input_shape[1:3]
st.write("🔍 شكل الإدخال المتوقع من النموذج:", input_shape)

# تحميل الصورة
uploaded_file = st.file_uploader("اختار صورة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="الصورة المُحمّلة", use_container_width=True)

    # تجهيز الصورة للتوقع
    img = img.resize(input_shape)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # التوقع
    prediction = model.predict(img_array)[0][0]
    health_score = (1 - prediction) * 100

    # عرض النتيجة
    st.write("## النتيجة:")
    st.write(f"🔍 نسبة صحة الورقة: {health_score:.2f}%")

    if health_score >= 75:
        st.success("✅ الورقة صحية")
    else:
        st.error("❌ الورقة غير صحية")
        st.warning("⚠️ رجاءً التحقق من الورقة، قد تحتاج إلى رعاية إضافية.")

        # ⬇️ تحليل الصورة باستخدام Roboflow
        st.subheader("📡 تحليل إضافي باستخدام Roboflow")

        # إعدادات Roboflow
        ROBOFLOW_API_KEY = "J5B3YIN1MZzAhnSQFLoV"  # ← استبدله بمفتاحك
        MODEL_ID = "strawberry-detections"
        VERSION = "1"

        api_url = f"https://detect.roboflow.com/{MODEL_ID}/{VERSION}?api_key={ROBOFLOW_API_KEY}"

        # تحويل الصورة لبايتس
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        buffered.seek(0)

        # إرسال الطلب
        response = requests.post(
            api_url,
            files={"file": buffered},
        )

        if response.status_code == 200:
            result = response.json()
            predictions = result.get("predictions", [])

            if len(predictions) == 0:
                st.warning("🤷‍♂️ لم يتم التعرف على أي مرض في الصورة.")
            else:
                first_prediction = predictions[0]
                disease_name = first_prediction.get("class", "غير معروف")
                confidence = first_prediction.get("confidence", 0) * 100

                st.success("✅ تم التعرف على المرض!")
                st.write(f"🦠 **نوع المرض:** {disease_name}")
                st.write(f"📊 **نسبة الثقة:** {confidence:.2f}%")

                # عرض معلومات المرض من القاموس
                disease_info = plant_diseases.get(disease_name)
                if disease_info:
                    st.subheader("📋 معلومات عن المرض:")
                    st.markdown(f"**🔹 السبب (Cause):** {disease_info['Cause']}")
                    st.markdown(f"**🧪 العلاج (Treatment):** {disease_info['Treatment']}")
                    st.markdown(f"**🛡️ الوقاية (Prevention):** {disease_info['Prevention']}")
                else:
                    st.info("ℹ️ لا توجد معلومات إضافية عن هذا المرض في قاعدة البيانات.")
        else:
            st.error("❌ فشل تحليل الصورة من Roboflow")
            st.text(response.text)
