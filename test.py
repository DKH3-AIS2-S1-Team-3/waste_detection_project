import requests

API_URL = "http://127.0.0.1:8000/detect"

# مهمة جداً: جيب أي صورة يكون فيها waste حسب التدريب
with open("test.jpg", "rb") as img:
    files = {"file": ("test.jpg", img, "image/jpeg")}
    r = requests.post(API_URL, files=files)

print("Status Code:", r.status_code)
print("Response:", r.json())