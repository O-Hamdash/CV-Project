import mediapipe as mp
import cv2


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6)


def get_face_landmarks(frame):
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    salient_points = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                salient_points.append((x, y))
    return salient_points


def get_landmarks_array(in_file):
    landmarks_array = []
    cap = cv2.VideoCapture(in_file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = get_face_landmarks(frame)
        landmarks_array.append(landmarks)
    
    return landmarks_array


def visualize_salient_points(in_file):
    cap = cv2.VideoCapture(in_file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = get_face_landmarks(frame)
        # Use landmarks for stabilization or visualization
        # Optionally visualize the landmarks on the frame
        print(landmarks)
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow('MediaPipe FaceMesh', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
