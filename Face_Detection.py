import cv2
import mediapipe as mp
from deepface import DeepFace
import time

def detect_emotion(face_image):
    # Convert to RGB
    rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    try:
        # Analyze the face for emotion, age, gender, and race
        analysis = DeepFace.analyze(
            rgb_face, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False
        )
        emotion = analysis[0]['dominant_emotion']
        gender = analysis[0]['dominant_gender']
        age = analysis[0]['age']
        race = analysis[0]['dominant_race']
        return f"{gender}, Age: {age}, {race}, {emotion}"
    except Exception as e:
        print(f"Error in detect_emotion: {e}")
        return "Unknown"


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def extract_faces_and_detect_emotions():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 1)
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x1 = int(bboxC.xmin * iw)
                    y1 = int(bboxC.ymin * ih)
                    x2 = x1 + int(bboxC.width * iw)
                    y2 = y1 + int(bboxC.height * ih)
                    face_image = frame[max(0, y1):min(
                        y2, ih), max(0, x1):min(x2, iw)]
                    if face_image.size == 0:
                        continue

                    for keypoint in detection.location_data.relative_keypoints:
                        keypoint_x = int(keypoint.x * iw)
                        keypoint_y = int(keypoint.y * ih)
                        cv2.circle(frame, (keypoint_x, keypoint_y),
                                   4, (0, 0, 255), -1)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Call detect_emotion and display attributes
                    attributes = detect_emotion(face_image)
                    cv2.putText(frame, attributes, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame in Colab
            cv2.imshow('image', frame)
            # time.sleep(2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# Run the function
extract_faces_and_detect_emotions()
