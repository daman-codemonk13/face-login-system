import cv2
from face_utils import encode_face, save_embedding

name = input("Enter your name: ")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("Register - Press S to Save", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f"images/{name}.jpg", frame)
        break

cap.release()
cv2.destroyAllWindows()

encoding = encode_face(f"images/{name}.jpg")
if encoding is not None:
    save_embedding(name, encoding)
    print(f"User {name} registered successfully!")
else:
    print("Face not detected. Try again.")
