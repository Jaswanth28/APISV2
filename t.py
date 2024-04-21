import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
import os
import csv
import datetime
import sqlite3

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin, you can change it to specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model('./BT_CNN_model_FINAL.h5')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
CSV_FILE = os.path.join(UPLOAD_FOLDER, 'records.csv')  # Path to the CSV file

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def predict_class(username, image):
    ci = ['glioma', 'meningioma', 'no-tumor', 'pituitary']

    image = cv2.resize(image, (176, 176))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    pci = np.argmax(prediction[0])
    predicted_class = ci[pci]
    current_datetime = datetime.datetime.now()
    date = current_datetime.date()
    time = current_datetime.strftime("%H:%M:%S.%f")[:-4]

    # Update the CSV file with the prediction result and username
    with open(CSV_FILE, mode='a', newline='') as csv_file:
        fieldnames = ['username', 'date', 'time', 'result']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Check if the CSV file is empty and write the header if necessary
        if os.path.getsize(CSV_FILE) == 0:
            writer.writeheader()
        
        writer.writerow({'username': username, 'date': date, 'time': time, 'result': predicted_class})
    try:
        conn = sqlite3.connect('./Data.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO history (username, date, time, result)
            VALUES (?, ?, ?, ?)
        ''', (username, date, time, predicted_class))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error inserting data into 'history' table: {e}")

    return predicted_class

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/upload")
async def upload_and_predict(username: str = Form(...), image: UploadFile = File(...)):
    if not allowed_file(image.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    predicted_class = predict_class(username, img)
    return JSONResponse(content={'predicted_class': predicted_class})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2000)
