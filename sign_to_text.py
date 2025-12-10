import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Define simple gestures (very basic examples)
gesture_dict = {
    "Thumbs Up": "👍 OK",
    "Peace": "✌ Peace"
}

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip image for mirror effect
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image
    results = hands.process(rgb_image)

    text_output = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Example: very simple rule-based detection
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Rule: if thumb is up
            if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y:
                text_output = gesture_dict["Thumbs Up"]

            # Rule: if index and middle fingers are raised
            elif index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y:
                text_output = gesture_dict["Peace"]

    # Show text on screen
    cv2.putText(image, text_output, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Show window
    cv2.imshow("Sign to Text", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()