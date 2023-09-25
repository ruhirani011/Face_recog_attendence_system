import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

camera_port = 0
video_capture = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)

newton_image = face_recognition.load_image_file("photos/newton.jpeg")
newton_encoding = face_recognition.face_encodings(newton_image)[0]

Ruhi_image = face_recognition.load_image_file("photos/Ruhi.jpeg")
Ruhi_encoding = face_recognition.face_encodings(Ruhi_image)[0]

anjali_image = face_recognition.load_image_file("photos/anjali.jpeg")
anjali_encoding = face_recognition.face_encodings(anjali_image)[0]

tata_image = face_recognition.load_image_file("photos/tata.jpeg")
tata_encoding = face_recognition.face_encodings(tata_image)[0]

sadmona_image = face_recognition.load_image_file("photos/sadmona.jpeg")
sadmona_encoding = face_recognition.face_encodings(sadmona_image)[0]

tesla_image = face_recognition.load_image_file("photos/tesla.jpeg")
tesla_encoding = face_recognition.face_encodings(tesla_image)[0]

known_face_encoding = [newton_encoding, Ruhi_encoding, anjali_encoding, tata_encoding, sadmona_encoding, tesla_encoding]

known_face_names = ["newton", "Ruhi", "anjali", "tata", "sadmona", "tesla"]

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    #print(students)
                    print(name)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

    cv2.imshow("Attendence_system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
f.close()


#cv2 takes input as bgr formate so we convert the frame into rgb frame

#It's important to note that with matplotlib, RGB is specifically for JPG images.
# PNG images will load as BGRA.

