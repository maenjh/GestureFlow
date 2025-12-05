import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, send_from_directory, request

app = Flask(__name__)

global_frame = None
output_dir = '/Users/moon/workspace/gestureflow/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def count_fingers(landmarks):
    """
    Counts the number of fingers that are open/extended.
    Returns a list of booleans [Thumb, Index, Middle, Ring, Pinky]
    indicating if each finger is open.
    """
    finger_tips = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky
    fingers_open = []

    # Handle Thumb (Check x-coordinate relative to the knuckle)
    # Assuming right hand for simplicity or mirroring logic. 
    # For a general webcam (mirrored), usually:
    # If tip x < ip x (for right hand on screen which is user's left if mirrored? No, let's use relative to wrist)
    # A simpler heuristic for thumb: Check if tip is farther from pinky base (17) than the joint (2) is.
    # Or just check x distance.
    
    # Let's use a simpler logic often used in tutorials:
    # For thumb, we check if the tip is to the "outside" of the IP joint.
    # Since we don't know hand handedness easily without checking label, 
    # we can check if tip.x < ip.x (for right hand) or tip.x > ip.x (for left hand).
    # But webcam is usually mirrored.
    
    # Let's rely on the standard logic:
    # Thumb: 4 vs 3 (IP). 
    # If x[4] > x[3] (Right hand facing camera) -> Open.
    # But let's try to be robust. Let's just check distance from wrist (0).
    # If dist(0, 4) > dist(0, 3) + threshold? No.
    
    # Let's stick to the standard vertical check for fingers and horizontal for thumb.
    # We will assume the hand is upright.
    
    # Thumb
    if landmarks[finger_tips[0]].x < landmarks[finger_tips[0] - 1].x:
        # This logic depends on hand side. Let's try to detect hand side or just ignore thumb for "Peace" and "Fist" vs "Palm" distinction if possible.
        # Actually, for "Fist", thumb is usually in. For "Palm", thumb is out.
        # Let's use a geometry based approach: Is thumb tip far from index mcp?
        pass
    
    # Let's use a simplified check for the 4 fingers first
    for id in range(1, 5): # Index to Pinky
        if landmarks[finger_tips[id]].y < landmarks[finger_tips[id] - 2].y:
            fingers_open.append(True)
        else:
            fingers_open.append(False)
            
    # For thumb, let's just check if it's extended. 
    # A simple way is to check if the tip is far from the index finger MCP (5).
    thumb_tip = landmarks[4]
    index_mcp = landmarks[5]
    dist = ((thumb_tip.x - index_mcp.x)**2 + (thumb_tip.y - index_mcp.y)**2)**0.5
    # Heuristic threshold. In normalized coords, hand is roughly 0.2-0.5 size.
    if dist > 0.15: # Adjust based on testing
        fingers_open.insert(0, True)
    else:
        fingers_open.insert(0, False)
        
    return fingers_open

def detect_gesture(fingers_open):
    """
    Determines the gesture based on open fingers.
    fingers_open: [Thumb, Index, Middle, Ring, Pinky]
    """
    # Palm: All 5 fingers open
    if all(fingers_open):
        return "Palm"
    
    # Fist: All fingers closed (or maybe thumb is tricky, so let's say at least 4 fingers closed)
    if not any(fingers_open) or (not any(fingers_open[1:]) and fingers_open[0]): 
        # 0 fingers open OR only thumb open (sometimes thumb detection is flaky)
        # Strict fist: all closed
        if not any(fingers_open):
            return "Fist"
        # Allow thumb to be open for fist if others are closed (common detection issue)
        if fingers_open[0] and not any(fingers_open[1:]):
            return "Fist"

    # Peace: Index and Middle open, others closed
    # fingers_open[1] is Index, [2] is Middle
    if fingers_open[1] and fingers_open[2] and not fingers_open[3] and not fingers_open[4]:
        return "Peace"
        
    return "Unknown"

def generate_frames():
    global global_frame
    cap = cv2.VideoCapture(0)
    
    # Capture state variables
    capture_start_time = None
    last_capture_time = 0
    capture_cooldown = 2.0
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from camera. Trying to reconnect...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(0)
            continue

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and find hands
        results = hands.process(rgb_frame)
        
        detected_gestures = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get landmarks list
                lm_list = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = frame.shape
                    lm_list.append(lm) # Keep normalized for relative checks
                
                # Count fingers
                fingers = count_fingers(lm_list)
                
                # Detect Gesture
                g = detect_gesture(fingers)
                detected_gestures.append(g)
        
        # Determine Filter Priority
        active_filter = "Normal"
        if "Peace" in detected_gestures:
            active_filter = "Sketch"
        elif "Fist" in detected_gestures:
            active_filter = "Grayscale"
        elif "Palm" in detected_gestures:
            active_filter = "Normal"
            
        # Apply Filters based on Active Gesture
        if active_filter == "Fist" or active_filter == "Grayscale": # Handle both naming conventions if needed, but we set it to Grayscale above
            # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) # Convert back to BGR so we can add colored text
            cv2.putText(frame, "Filter: Grayscale", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
        elif active_filter == "Sketch":
            # Canny Edge Detection (Sketch)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            # Invert to make it look like sketch on white
            edges = cv2.bitwise_not(edges)
            frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.putText(frame, "Filter: Sketch", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # Overlay Gesture Names
        gesture_text = ", ".join(detected_gestures) if detected_gestures else "None"
        cv2.putText(frame, f"Gestures: {gesture_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Update global frame for manual capture
        global_frame = frame.copy()

        # Capture Logic (Fist Trigger)
        if "Fist" in detected_gestures:
            current_time = time.time()
            if current_time - last_capture_time > capture_cooldown:
                if capture_start_time is None:
                    capture_start_time = current_time
                
                elapsed = current_time - capture_start_time
                remaining = 3 - int(elapsed)
                
                if remaining > 0:
                    # Draw Countdown
                    h, w, _ = frame.shape
                    cv2.putText(frame, str(remaining), (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5)
                else:
                    # Capture
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(output_dir, f"capture_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"Saved {filename}")
                    
                    # Visual Feedback
                    h, w, _ = frame.shape
                    cv2.putText(frame, "Captured!", (w//2 - 150, h//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                    
                    capture_start_time = None
                    last_capture_time = current_time
        else:
            capture_start_time = None

        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    global global_frame
    if global_frame is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, global_frame)
        return jsonify({'status': 'success', 'filename': filename})
    return jsonify({'status': 'error', 'message': 'No frame available'}), 400

@app.route('/gallery')
def gallery():
    images = []
    if os.path.exists(output_dir):
        # Get list of files, sorted by modification time (newest first)
        files = [f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
        images = files
    return jsonify(images)

@app.route('/output/<path:filename>')
def serve_image(filename):
    return send_from_directory(output_dir, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
