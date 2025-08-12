from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
import os
from flask_cors import CORS


def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    image_folder = 'img'
    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_folder, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])


app = Flask(__name__)
CORS(app)

# Cargar rostros conocidos
image_folder = 'img'

image_paths = [
    os.path.join(image_folder, filename)
    for filename in os.listdir(image_folder)
    if filename.lower().endswith((".png", ".jpg", ".jpeg"))
]
known_face_encodings = []
known_face_names = []

for image_path in image_paths:
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_face_encodings.append(encodings[0])
        known_face_names.append(os.path.basename(image_path).split('.')[0])

@app.route('/recognize', methods=['POST'])
def recognize():
    #load_known_faces()
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=0.5)
        if True in matches:
            idx = matches.index(True)
            name = known_face_names[idx]
            print(f"🔵 Reconocido: {name}")  # <-- Agrega este log
            return jsonify({'name': name})

    return jsonify({'name': 'Desconocido'})

@app.route('/register', methods=['POST'])
def register():
    print("🔵 Recibiendo petición en /register")
    print("🔵 Archivos recibidos:", request.files)
    print("🔵 Formulario recibido:", request.form)

    file = request.files['image']
    name = request.form['name']

    if not file:
        print("❌ No se recibió el archivo de imagen")
        return jsonify({'message': 'No se recibió el archivo de imagen'}), 400

    if not name:
        print("❌ No se recibió el nombre")
        return jsonify({'message': 'No se recibió el nombre'}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_frame)

    if encodings:
        fileName = f"{name}.jpg"
        save_path = os.path.join(image_folder, fileName)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, frame)

        known_face_encodings.append(encodings[0])
        known_face_names.append(name)
        return jsonify({'message': 'Rostro registrado exitosamente'})
    else:
        return jsonify({'message': 'No se pudo detectar un rostro en la imagen'}), 400

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'active',
        'message': 'Servidor de reconocimiento facial funcionando'
    })

if __name__ == '__main__':
    app.run(port=5001)
