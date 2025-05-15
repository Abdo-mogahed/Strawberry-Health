# استخدم صورة رسمية من Python
FROM python:3.9-slim

# تعيين مجلد العمل داخل الحاوية
WORKDIR /app

# نسخ الملفات إلى الحاوية
COPY . .

# تثبيت المتطلبات
RUN pip install --no-cache-dir -r requirements.txt

# فتح المنفذ الافتراضي لستريملت
EXPOSE 8501

# الأمر الذي يشغل التطبيق
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
