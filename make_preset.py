import cv2
from calibration import CheckerboardCalibrator

# Change this to the exact physical width of ONE square on your printed board.
SQUARE_PHYSICAL_SIZE_CM = 2.84 
DISTANCE = 5

# The image containing the checkerboard you want to calibrate against
IMAGE_PATH = "images/test_A_1.JPG" 

def generate_preset():
    print(f"Loading visual source: {IMAGE_PATH}")
    frame = cv2.imread(IMAGE_PATH)
    
    if frame is None:
        print(f"Error: Could not load {IMAGE_PATH}.")
        return

    # Initialize the calibrator from cal2.py
    # We pass the physical size from our configuration above
    calibrator = CheckerboardCalibrator(pattern_size=(9, 6), square_size_cm=SQUARE_PHYSICAL_SIZE_CM)

    print("Searching for checkerboard...")
    
    # Process the frame using our imported class
    ppm = calibrator.process(frame)

    if ppm is not None:
        print("Calibration Successful")
        print(f"Physical Square Size Configured: {SQUARE_PHYSICAL_SIZE_CM} cm")
        print(f"Calibrated Preset Value (PPM): {ppm:.4f} px/cm")
        print(f"Distance': {DISTANCE} ft")
        
        # We can still show the frame because calibrator.process() draws the corners onto it
        cv2.imshow("Calibration Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("\Calibration Failed: No valid checkerboard found in the image.")
        print("Ensure the board is visible, flat, upright, and has a white margin.")

if __name__ == "__main__":
    generate_preset()
