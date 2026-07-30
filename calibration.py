import cv2
import math

class CheckerboardCalibrator:
    def __init__(self, pattern_size=(9, 6), square_size_cm=2.5):
        self.pattern_size = pattern_size 
        self.square_size_cm = square_size_cm 

    def _calculate_distance(self, p1, p2):
        return math.sqrt((float(p2[0]) - float(p1[0]))**2 + (float(p2[1]) - float(p1[1]))**2)

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a blur to the grayscale image. 
        # This smooths out small smudges, noise, and sharp lighting artifacts
        # so the algorithm only focuses on the large, dominant squares.
        blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Try to find the corners on the blurred image
        ret, corners = cv2.findChessboardCorners(blurred_gray, self.pattern_size, 
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                 cv2.CALIB_CB_FAST_CHECK + 
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # We refine the corners on the original crisp grayscale image, not the blurred one
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            cv2.drawChessboardCorners(frame, self.pattern_size, corners_refined, ret)

            flat_corners = corners_refined.reshape(-1, 2)
            
            point1 = flat_corners[0]
            point2 = flat_corners[1]
            
            pixel_width = self._calculate_distance(point1, point2)

            # Safety check. If the detected square is suspiciously tiny (e.g., less than 5 pixels wide),
            # it means OpenCV hallucinated a grid on a smudge. Reject it.
            if pixel_width < 5.0:
                return None

            ppm = pixel_width / self.square_size_cm
            return ppm
            
        return None 