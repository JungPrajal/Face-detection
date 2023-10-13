import pathlib
import cv2

#Cascade XML file for face detection
cascade_path = pathlib.Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'

# Create a CascadeClassifier
clf = cv2.CascadeClassifier(str(cascade_path))

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 0), 2)

    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord("q"):
        break

# Release the camera and destroy all OpenCV windows
camera.release()
cv2.destroyAllWindows()
