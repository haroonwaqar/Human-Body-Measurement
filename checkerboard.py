import cv2
import numpy as np

# This generates a 10x7 grid of squares.
# This yields 9 internal horizontal corners and 6 internal vertical corners.
# This matches the pattern_size=(9, 6) we hardcoded into calibration.py
SQUARES_X = 10
SQUARES_Y = 7
PIXELS_PER_SQUARE = 200 # Increased for higher print quality

def create_checkerboard():
    width = SQUARES_X * PIXELS_PER_SQUARE
    height = SQUARES_Y * PIXELS_PER_SQUARE
    
    # Create an image completely filled with white pixels (value 255)
    board = np.ones((height, width), dtype=np.uint8) * 255

    # Iterate through the grid and draw the black squares
    for y in range(SQUARES_Y):
        for x in range(SQUARES_X):
            # If the sum of coordinates is even, color the square black
            if (x + y) % 2 == 0:
                start_x = x * PIXELS_PER_SQUARE
                start_y = y * PIXELS_PER_SQUARE
                # Set the pixel values in this square area to 0 (black)
                board[start_y:start_y+PIXELS_PER_SQUARE, start_x:start_x+PIXELS_PER_SQUARE] = 0

    # Add a white border around the entire board. 
    # CRITICAL: OpenCV cannot detect the outer corners if there is no white space around them.
    border_size = PIXELS_PER_SQUARE
    board_with_border = cv2.copyMakeBorder(board, border_size, border_size, border_size, border_size, 
                                           cv2.BORDER_CONSTANT, value=255)

    filename = f"checkerboard_{SQUARES_X}x{SQUARES_Y}.png"
    cv2.imwrite(filename, board_with_border)
    
    print(f"Successfully generated {filename}.")

if __name__ == "__main__":
    create_checkerboard()