import cv2
import mediapipe as mp
import pyautogui

# Initialize the webcam
cam = cv2.VideoCapture(0)

# Initialize MediaPipe Face Mesh solution with refined landmarks
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get screen width and height for controlling the mouse pointer
screen_w, screen_h = pyautogui.size()

# Infinite loop to process each frame from the webcam
while True:
    # Capture a frame from the webcam
    _, frame = cam.read()

    # Flip the frame horizontally for a more natural interaction
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame to detect facial landmarks
    output = face_mesh.process(rgb_frame)

    # Get the list of facial landmarks detected
    landmark_points = output.multi_face_landmarks

    # Get the dimensions of the frame
    frame_h, frame_w, _ = frame.shape

    # Check if any landmarks are detected
    if landmark_points:
        # Extract the first face's landmarks
        landmarks = landmark_points[0].landmark

        # Loop through specific eye landmarks (id 474 to 478) for cursor control
        for id, landmark in enumerate(landmarks[474:478]):
            # Calculate the (x, y) coordinates in the frame
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)

            # Draw a small circle at each landmark point
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

            # If the landmark id is 1, use it to move the mouse pointer
            if id == 1:
                # Map the frame coordinates to screen coordinates
                screen_x = screen_w / frame_w * x
                screen_y = screen_h / frame_h * y

                # Move the mouse pointer to the calculated screen coordinates
                pyautogui.moveTo(screen_x, screen_y)

        # Get landmarks for the left eye to detect blinking (id 145 and 159)
        left = [landmarks[145], landmarks[159]]

        # Draw circles at the left eye landmarks
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        # Check if the eye is closed (y distance between landmarks is small)
        if (left[0].y - left[1].y) < 0.004:
            # Perform a mouse click action
            pyautogui.click()
            # Sleep for 1 second to avoid multiple clicks
            pyautogui.sleep(1)

    # Display the frame with landmarks
    cv2.imshow('eye controlled mouse', frame)

    # Wait for a short period before processing the next frame
    cv2.waitKey(1)
