import cv2
import mediapipe as mp

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for natural mirroring
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB (Mediapipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with Mediapipe
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmark positions
                landmarks = hand_landmarks.landmark

                # Define finger tip landmarks (Thumb, Index, Middle, Ring, Pinky)
                finger_tips = [4, 8, 12, 16, 20]
                finger_count = 0

                # Get wrist and index finger base as reference
                wrist_x = landmarks[0].x
                index_base_x = landmarks[5].x  # Base of index finger

                # Count fingers (excluding thumb)
                for tip in finger_tips[1:]:  # Skip thumb for now
                    if landmarks[tip].y < landmarks[tip - 2].y:  # Finger is raised
                        finger_count += 1

                # Thumb Detection - Compare to wrist & index base
                thumb_tip_x = landmarks[4].x
                thumb_base_x = landmarks[2].x  # Base of thumb

                if wrist_x < index_base_x:  # Right hand
                    if thumb_tip_x > thumb_base_x:  # Thumb is extended
                        finger_count += 1
                else:  # Left hand
                    if thumb_tip_x < thumb_base_x:  # Thumb is extended
                        finger_count += 1

                # Display finger count
                cv2.putText(frame, f'Fingers: {finger_count}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show frame
        cv2.imshow("Hand & Finger Detection", frame)

        # Check if window is closed
        if cv2.getWindowProperty("Hand & Finger Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
