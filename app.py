import cv2
import face_recognition

try:
    image_paths = ['img/wilber.png', 'img/paola.png']
    known_face_encodings = []
    known_face_names = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f'No se encontró la imagen: {image_path}')
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        for encoding in face_encodings:
            known_face_encodings.append(encoding)
            known_face_names.append(image_path)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError('No se pudo abrir la cámara.')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (face_location, face_encoding) in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Desconocido"
            color = (50, 50, 255)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                color = (125, 228, 0)

            cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), color, 2)
            cv2.putText(frame, name, (face_location[3], face_location[2] + 20), 2, 0.7, (255, 255, 255), 1)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()