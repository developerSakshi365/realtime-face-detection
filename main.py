import cv2

# Load the pre-trained Haar cascade for face detection
face_cap = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")


# Start capturing video from webcam (0 = default camera)
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()
    if not ret:
        break  # camera not available or failed to read

    # Convert the frame to grayscale
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw green rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the video frame
    cv2.imshow("Video_live", video_data)

    # Press 'a' key to exit the loop
    if cv2.waitKey(10) == ord("a"):
        break

# Release the webcam and close the window
video_cap.release()
cv2.destroyAllWindows()
