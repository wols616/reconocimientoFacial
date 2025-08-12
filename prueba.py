import cv2
import face_recognition

try:
    image = cv2.imread('img/fotoPrueba.png')
    if image is None:
        raise FileNotFoundError('No se encontró la imagen.')

    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        raise ValueError('No se detectaron caras en la imagen.')

    face_image_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
    if not face_image_encodings:
        raise ValueError('No se pudieron codificar las caras.')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError('No se pudo abrir la cámara.')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        face_locations = face_recognition.face_locations(frame)
        face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations=face_locations)

        for (face_location, face_encoding) in zip(face_locations, face_frame_encodings):
            result = face_recognition.compare_faces(face_image_encodings, face_encoding, tolerance=0.6)
            print(result)
            text = "Wil" if any(result) else "Sepa quien es"
            color = (125, 228, 0) if any(result) else (50, 50, 255)

            cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), color, 2)
            cv2.putText(frame, text, (face_location[3], face_location[2] + 20), 2, 0.7, (255, 255, 255), 1)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()