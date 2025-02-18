import cv2
import mediapipe as mp

#inizialization mediapipe mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

#Landmarks indexes for both eyes (MediaPipe FaceMesh)
LEFT_EYE_LANDMARKS = [33, 133, 160, 158, 144, 153, 362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [362, 263, 387, 385, 373, 380, 33, 133, 160, 158, 144, 153]

#start the webcam
cap = cv2.VideoCapture(0)


#while loop to capture the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    #converts the image taken by the webcam in RGB cause opencv works with BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #face mesh process the image with the key face landmarks
    results = face_mesh.process(rgb_frame)
    
    
    #if the process find face lendmarks it enter in the if cycle(it found a face)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            #inizialization of the key coordinates of the eyes
            h, w, _ = frame.shape
            left_eye_pts = []
            right_eye_pts = []
            
            #it access the normalized position of every landmark in right and left eye, it converts the norm positions in pixel positions in the frame and then color them 
            for idx in LEFT_EYE_LANDMARKS:
                x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                left_eye_pts.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            for idx in RIGHT_EYE_LANDMARKS:
                x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                right_eye_pts.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            #drawing the square around the eyes
            if left_eye_pts and right_eye_pts:
                left_x_min, left_y_min = min(left_eye_pts, key=lambda p: p[0])[0], min(left_eye_pts, key=lambda p: p[1])[1]
                left_x_max, left_y_max = max(left_eye_pts, key=lambda p: p[0])[0], max(left_eye_pts, key=lambda p: p[1])[1]
                right_x_min, right_y_min = min(right_eye_pts, key=lambda p: p[0])[0], min(right_eye_pts, key=lambda p: p[1])[1]
                right_x_max, right_y_max = max(right_eye_pts, key=lambda p: p[0])[0], max(right_eye_pts, key=lambda p: p[1])[1]
                
                cv2.rectangle(frame, (left_x_min, left_y_min), (left_x_max, left_y_max), (255, 0, 0), 2)
                cv2.rectangle(frame, (right_x_min, right_y_min), (right_x_max, right_y_max), (255, 0, 0), 2)
    
    #shows the video
    cv2.imshow('Eye Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
