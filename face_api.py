from flask import Flask, request, jsonify, send_file
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

# Carpeta donde se guardan las imágenes
image_folder = 'img'
os.makedirs(image_folder, exist_ok=True)

# Cargar rostros conocidos al inicio
load_known_faces()


def get_existing_photo_path(name):
    """Busca la foto de una persona con cualquier extensión válida"""
    for ext in [".jpg", ".jpeg", ".png"]:
        file_path = os.path.join(image_folder, f"{name}{ext}")
        if os.path.exists(file_path):
            return file_path
    return None


@app.route('/recognize', methods=['POST'])
def recognize():
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
            print(f"🔵 Reconocido: {name}")
            return jsonify({'name': name})

    return jsonify({'name': 'Desconocido'})


@app.route('/register', methods=['POST'])
def register():
    file = request.files['image']
    name = request.form['name']

    if not file or not name:
        return jsonify({'message': 'Faltan datos (imagen o nombre)'}), 400

    # Detectar extensión original
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        return jsonify({'message': 'Formato de imagen no soportado'}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_frame)

    if encodings:
        save_path = os.path.join(image_folder, f"{name}{ext}")
        cv2.imwrite(save_path, frame)

        load_known_faces()
        return jsonify({'message': f'Rostro registrado exitosamente como {save_path}'})
    else:
        return jsonify({'message': 'No se pudo detectar un rostro en la imagen'}), 400


@app.route('/update', methods=['POST'])
def update():
    """
    Reemplazar la imagen de una persona existente
    """
    file = request.files['image']
    name = request.form['name']

    if not file or not name:
        return jsonify({'message': 'Faltan datos (imagen o nombre)'}), 400

    # Verificar si existe alguna foto previa
    old_path = get_existing_photo_path(name)
    if not old_path:
        return jsonify({'message': 'La persona no está registrada'}), 404

    # Detectar extensión nueva
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        return jsonify({'message': 'Formato de imagen no soportado'}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_frame)

    if encodings:
        # Borrar la foto anterior
        os.remove(old_path)
        # Guardar con nueva extensión
        new_path = os.path.join(image_folder, f"{name}{ext}")
        cv2.imwrite(new_path, frame)

        load_known_faces()
        return jsonify({'message': 'Fotografía actualizada correctamente'})
    else:
        return jsonify({'message': 'No se pudo detectar un rostro en la nueva imagen'}), 400


@app.route('/delete', methods=['POST'])
def delete():
    """
    Eliminar la imagen de una persona
    """
    name = request.form['name']

    if not name:
        return jsonify({'message': 'Debe proporcionar el nombre'}), 400

    file_path = get_existing_photo_path(name)

    if file_path:
        os.remove(file_path)
        load_known_faces()
        return jsonify({'message': 'Fotografía eliminada correctamente'})
    else:
        return jsonify({'message': 'No se encontró la fotografía de esa persona'}), 404


@app.route('/photo-exists/<name>', methods=['GET'])
def photo_exists(name):
    file_path = get_existing_photo_path(name)
    return jsonify({'exists': file_path is not None})


@app.route('/get-photo/<name>', methods=['GET'])
def get_photo(name):
    file_path = get_existing_photo_path(name)
    if file_path:
        # Ajustar el mimetype según la extensión
        ext = os.path.splitext(file_path)[1].lower()
        mimetype = 'image/jpeg' if ext in [".jpg", ".jpeg"] else 'image/png'
        return send_file(file_path, mimetype=mimetype)
    else:
        return jsonify({'message': 'Foto no encontrada'}), 404

@app.route('/rename', methods=['POST'])
def rename():
    """
    Cambiar el nombre de una persona (y su foto asociada)
    """
    old_name = request.form.get('old_name')
    new_name = request.form.get('new_name')

    if not old_name or not new_name:
        return jsonify({'message': 'Debe proporcionar old_name y new_name'}), 400

    # Buscar foto existente
    old_path = get_existing_photo_path(old_name)
    if not old_path:
        return jsonify({'message': f'No se encontró la fotografía de {old_name}'}), 404

    # Obtener extensión
    ext = os.path.splitext(old_path)[1].lower()
    new_path = os.path.join(image_folder, f"{new_name}{ext}")

    # Verificar que no exista ya un archivo con el nuevo nombre
    if os.path.exists(new_path):
        return jsonify({'message': 'Ya existe una persona con ese nuevo nombre'}), 400

    # Renombrar el archivo
    os.rename(old_path, new_path)

    # Recargar rostros
    load_known_faces()

    return jsonify({'message': f'Nombre cambiado de {old_name} a {new_name} exitosamente'})


@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'active',
        'message': 'Servidor de reconocimiento facial funcionando'
    })


if __name__ == '__main__':
    app.run(port=5001)
