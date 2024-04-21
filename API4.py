import sqlite3
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from flask_cors import CORS
import cv2
import os
import csv
import datetime


app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model('./BT_CNN_model_FINAL.h5')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
CSV_FILE = os.path.join(UPLOAD_FOLDER, 'records.csv')  # Path to the CSV file

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_history_records(username):
    records = []
    try:
        conn = sqlite3.connect('./user_credentials.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT date, time, result
            FROM history
            WHERE username = ?
        ''', (username,))
        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            records.append({
                'date': row[0],
                'time': row[1],
                'result': row[2]
            })
    except Exception as e:
        print(f"Error fetching history records: {e}")

    return records

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

@app.route("/history", methods=['GET'])
def get_user_history():
    username = request.args.get('username')  # Get the username from the query parameter
        # Fetch history records for the username
    records = get_history_records(username)
    if not records:
        return ('', 204)

    return jsonify(records)

@app.route("/upload", methods=['POST'])
def upload_and_predict():
    username = request.form.get('username')
    if 'image' not in request.files:
        return jsonify({'predicted_class': ''})  

    file = request.files['image']

    if file.filename == '':
        return jsonify({'predicted_class': ''})  

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Load the saved image
        image = cv2.imread(filename)

        # Perform prediction with username
        predicted_class = predict_class(username, image)

        # Delete the uploaded image after prediction if needed
        os.remove(filename)
        return jsonify({'predicted_class': predicted_class})

    return jsonify({'predicted_class': ''})  

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=2000)
