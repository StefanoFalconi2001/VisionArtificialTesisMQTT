import pickle
import cv2
import mediapipe as mp
import numpy as np
import easymodbus.modbusClient

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the Modbus client
modbus_client = easymodbus.modbusClient.ModbusClient('192.168.178.110', 502)
modbus_client.connect()

# Initialize the video capture (0 = default webcam)
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label dictionary for predictions
labels_dict = {0: 'Start', 1: 'Stop', 2: 'Clockwise spin', 3: 'Counter clockwise spin', 4: 'Increase speed', 5: 'Decrease speed'}

# State variables
is_running = False
speed_level = 0
max_speed_level = 7
clockwise_executed = False
counter_clockwise_executed = False

# Function to send data to PLC based on gesture
def send_to_plc(register_value):
    modbus_client.write_single_register(0, register_value)

def handle_gesture(predicted_character, frame):
    global is_running, speed_level, clockwise_executed, counter_clockwise_executed
    error_message = None

    if predicted_character == 'Start':
        if not is_running:
            send_to_plc(1)  # Start command
            is_running = True
        else:
            error_message = "Cannot execute 'Start' again until 'Stop' is executed."

    elif predicted_character == 'Stop':
        send_to_plc(2)  # Stop command
        is_running = False
        speed_level = 0
        clockwise_executed = False
        counter_clockwise_executed = False
        print("Stop gesture detected. Shutting down system...")
        cap.release()
        cv2.destroyAllWindows()
        modbus_client.close()
        exit()

    elif predicted_character == 'Clockwise spin':
        if not clockwise_executed:
            send_to_plc(3)  # Clockwise spin command
            clockwise_executed = True
        else:
            error_message = "Cannot execute 'Clockwise spin' again."

    elif predicted_character == 'Counter clockwise spin':
        if not counter_clockwise_executed:
            send_to_plc(4)  # Counter clockwise spin command
            counter_clockwise_executed = True
        else:
            error_message = "Cannot execute 'Counter clockwise spin' again."

    elif predicted_character == 'Increase speed':
        if is_running and speed_level < max_speed_level:
            send_to_plc(5)  # Increase speed command
            speed_level += 1
        else:
            error_message = "Maximum speed reached or system not running."

    elif predicted_character == 'Decrease speed':
        if is_running and speed_level > 0:
            send_to_plc(6)  # Decrease speed command
            speed_level -= 1
        else:
            error_message = "Minimum speed reached or system not running."

    if error_message:
        cv2.putText(frame, error_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        handle_gesture(predicted_character, frame)  # Handle gesture with validation and error display

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key (ASCII 27)
        print("ESC key pressed. Exiting...")
        break

    if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed manually. Exiting...")
        break

# Cleanup code (should not reach here if Stop gesture is handled correctly)
cap.release()
cv2.destroyAllWindows()
modbus_client.close()

