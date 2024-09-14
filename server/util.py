import joblib
import json
import numpy as np
import base64
import cv2
import pywt  # Use PyWavelets instead of a custom wavelet module
import os
# Global variables to store mappings and model
__class_name_to_number = {}
__class_number_to_name = {}

__model = None

# Wavelet transform function
def w2d(img, mode='db1', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray) / 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H = np.uint8(imArray_H * 255)
    return imArray_H

# Classification function with error handling
def classify_image(image_base64_data, file_path=None):
    try:
        # Process image from either file path or base64 data
        imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
        if not imgs:
            return {"error": "No valid faces detected in the image"}

        result = []
        for img in imgs:
            # Resize and preprocess the raw image
            scaled_raw_img = cv2.resize(img, (32, 32))
            img_har = w2d(img, 'db1', 5)
            scaled_img_har = cv2.resize(img_har, (32, 32))
            
            # Combine raw and transformed images into a feature vector
            combined_img = np.vstack((
                scaled_raw_img.reshape(32 * 32 * 3, 1),
                scaled_img_har.reshape(32 * 32, 1)
            ))
            len_image_array = 32*32*3 + 32*32
            
            final = combined_img.reshape(1, len_image_array).astype(float)
            
            # Predict class and probabilities
            class_prediction = __model.predict(final)[0]
            class_probabilities = np.around(__model.predict_proba(final) * 100, 2).tolist()[0]
            
            result.append({
                'class': class_number_to_name(class_prediction),
                'class_probability': class_probabilities,
                'class_dictionary': __class_name_to_number
            })
        return result

    except Exception as e:
        return {"error": str(e)}


# Function to get class name from class number
def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

# Load artifacts (model and class dictionary)
def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name
     # Get the absolute path to the current script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'class_dictionary.json')
    # Load the class dictionary
    with open(file_path, "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    # Load the model
    global __model
    if __model is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'saved_model.pkl')
        with open(file_path , 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")

# Convert base64 string to OpenCV image
def get_cv2_image_from_base64_string(b64str):
    if isinstance(b64str, bytes):
        b64str = b64str.decode('utf-8')
    if ',' in b64str:
        encoded_data = b64str.split(',')[1]
    else:
        encoded_data = b64str

    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# Function to detect faces and ensure 2 eyes are detected before cropping
def get_cropped_image_if_2_eyes(image_path=None, image_base64_data=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    face_cascade_path = os.path.join(base_dir, 'haarcascade', 'haarcascade_frontalface_default.xml')
    eye_cascade_path = os.path.join(base_dir, 'haarcascade', 'haarcascade_eye.xml')
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    if face_cascade.empty():
        raise Exception("Error loading face cascade.")
    if eye_cascade.empty():
        raise Exception("Error loading eye cascade.")

    if image_path:
        file_path = os.path.join(base_dir, image_path)
        img = cv2.imread(file_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cropped_faces = []
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    
    return cropped_faces

# Load base64 image for testing
def get_b64_test_image():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'b64.txt')
    with open(file_path) as f:
        return f.read()

# Main method to test locally
if __name__ == '__main__':
    load_saved_artifacts()
    # print(classify_image(get_b64_test_image()))
    print(classify_image(None, "./test_images/1fee075645.jpg"))
