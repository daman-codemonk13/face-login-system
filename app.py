import cv2
from face_utils import encode_face, load_all_embeddings, match_face

known_embeddings = load_all_embeddings()
cap = cv2.VideoCapture(0)

print("Looking for face...")

while True:
    ret, frame = cap.read()
    cv2.imshow("Login - Press L to Login", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('l'):
        cv2.imwrite("live.jpg", frame)
        break

cap.release()
cv2.destroyAllWindows()

face_encoding = encode_face("live.jpg")
if face_encoding is not None:
    name = match_face(face_encoding, known_embeddings)
    if name:
        print(f"✅ Login Successful. Welcome, {name}!")
    else:
        print("❌ Face not recognized.")
else:
    print("No face detected.")
