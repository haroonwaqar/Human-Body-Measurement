import cv2
import math
from ultralytics import YOLO

# calculate these values manually once using your camera and the checkerboard
# Format: { "Distance Name": PPM_Value }
# Example: 20.5 means 20.5 pixels equal 1 centimeter at that specific distance.
CALIBRATION_PRESETS = {
    "4_feet": 25.0204, 
    "5_feet": 21.1071,
    "6_feet": 16.2010,
    "8_feet": 11.1325,
}

# Select which preset you want to use for the current execution
ACTIVE_PRESET = "5_feet" 

# Set to '0' for live webcam, or keep as a string for an image path
VISUAL_SOURCE = 0 #"images/test_A_1.JPG" 

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def draw_measurement(frame, p1, p2, text, color, offset=(0,0)):
    cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)
    mid_x = int((p1[0] + p2[0]) / 2) + offset[0]
    mid_y = int((p1[1] + p2[1]) / 2) + offset[1]
    cv2.putText(frame, text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def process_frame(frame, model, ppm):
    # Display active configuration on screen
    cv2.putText(frame, f"Preset: {ACTIVE_PRESET} ({ppm:.2f} px/cm)", 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    results = model(frame, device="mps", verbose=False)

    for r in results:
        if r.keypoints is not None and len(r.keypoints.xy[0]) > 0:
            kp = r.keypoints.xy[0].cpu().numpy()
            
            if len(kp) >= 17:
                # Shoulders
                if kp[5][0] != 0 and kp[6][0] != 0:
                    dist_px = calculate_distance(kp[5], kp[6])
                    draw_measurement(frame, kp[5], kp[6], f"Shoulders: {dist_px/ppm:.1f}cm", (0, 255, 255), (0, -20))

                # Right Arm
                if kp[6][0] != 0 and kp[8][0] != 0 and kp[10][0] != 0:
                    total_arm_cm = (calculate_distance(kp[6], kp[8]) + calculate_distance(kp[8], kp[10])) / ppm
                    cv2.line(frame, (int(kp[6][0]), int(kp[6][1])), (int(kp[8][0]), int(kp[8][1])), (255, 0, 255), 2)
                    draw_measurement(frame, kp[8], kp[10], f"R Arm: {total_arm_cm:.1f}cm", (255, 0, 255), (20, 0))

                # Left Arm
                if kp[5][0] != 0 and kp[7][0] != 0 and kp[9][0] != 0:
                    total_arm_cm = (calculate_distance(kp[5], kp[7]) + calculate_distance(kp[7], kp[9])) / ppm
                    cv2.line(frame, (int(kp[5][0]), int(kp[5][1])), (int(kp[7][0]), int(kp[7][1])), (255, 0, 255), 2)
                    draw_measurement(frame, kp[7], kp[9], f"L Arm: {total_arm_cm:.1f}cm", (255, 0, 255), (-120, 0))

                # Estimated Height
                if kp[0][0] != 0 and kp[15][0] != 0 and kp[16][0] != 0:
                    nose = kp[0]
                    center_ankle_x = (kp[15][0] + kp[16][0]) / 2
                    center_ankle_y = (kp[15][1] + kp[16][1]) / 2
                    center_ankle = (center_ankle_x, center_ankle_y)
                    
                    nose_to_floor_px = calculate_distance(nose, center_ankle)
                    skull_buffer_px = 12.0 * ppm 
                    total_height_px = nose_to_floor_px + skull_buffer_px
                    
                    height_cm = total_height_px / ppm
                    draw_measurement(frame, nose, center_ankle, f"Height: {height_cm:.1f}cm", (0, 255, 0), (-250, -300))
    return frame

def main():
    # 1. Validation
    ppm = CALIBRATION_PRESETS.get(ACTIVE_PRESET)
    if ppm is None:
        print(f"Error: Preset '{ACTIVE_PRESET}' not found in configuration.")
        return

    print("Loading YOLOv11 Pose Model on MPS...")
    model = YOLO("yolo11n-pose.pt")

    # 2. Handle Video Source (Image or Live Stream)
    if isinstance(VISUAL_SOURCE, str):
        # Processing a single static image
        print(f"Processing static image: {VISUAL_SOURCE}")
        frame = cv2.imread(VISUAL_SOURCE)
        if frame is None:
            print("Error: Could not load image.")
            return
            
        processed_frame = process_frame(frame, model, ppm)

        output_filename = "output_full_measurements.jpg"
        cv2.imwrite(output_filename, processed_frame)
        print(f"Saved to {output_filename}")
        
        
        # Display logic for image
        h, w = processed_frame.shape[:2]
        if w > 1000:
            processed_frame = cv2.resize(processed_frame, (1000, int(h * (1000/w))))
        cv2.imshow("Measurement System", processed_frame)
        cv2.waitKey(0) 
        
    else:
        # Processing a live webcam or ESP32 stream
        print(f"Opening video stream source: {VISUAL_SOURCE}")
        cap = cv2.VideoCapture(VISUAL_SOURCE)
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return
            
        while True:
            success, frame = cap.read()
            if not success:
                break
                
            processed_frame = process_frame(frame, model, ppm)
            cv2.imshow("Measurement System (Live)", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
