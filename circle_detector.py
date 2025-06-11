import numpy as np
import cv2

class CircleDetector:
    def __init__(self, num_rois=6, min_score=0.7):
        """
        Initialize the CircleDetector class.
        
        Args:
            num_rois (int): Number of ROIs to generate
            min_score (float): Minimum score threshold for accepting a circle (0-1)
        """
        self.image = None
        self.processed_image = None
        self.result = None
        self.num_rois = num_rois
        self.min_score = min_score

    def load_image(self, image_path):
        """Load and store the image."""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        return self

    def preprocess_image(self):
        """Preprocess the image using HSV color space and create mask."""
        if self.image is None:
            raise ValueError("No image loaded")

        # Convert to HSV and enhance saturation
        hsv_frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_frame)
        s = np.clip(s * 1.5, 0, 255).astype(np.uint8)
        hsv_enhanced = cv2.merge([h, s, v])

        # Create color mask
        lower_bound = np.array([5, 220, 0])
        upper_bound = np.array([11, 255, 255])
        mask = cv2.inRange(hsv_enhanced, lower_bound, upper_bound)

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.medianBlur(mask, 23)
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)

        self.processed_image = mask
        return self

    def _get_random_rois(self, points):
        """
        Generate random ROIs based on point distribution.
        
        Args:
            points: Array of (x,y) coordinates
            
        Returns:
            List of (roi_img, (x,y)) tuples
        """
        xs, ys = points[:, 0], points[:, 1]
        min_x, min_y = np.min(xs), np.min(ys)
        max_x, max_y = np.max(xs), np.max(ys)
        width = max_x - min_x
        height = max_y - min_y

        rois = []
        for _ in range(self.num_rois):
            # ROI size (70%-100% of point spread)
            roi_width = np.random.randint(int(width * 0.7), int(width * 1.0))
            roi_height = np.random.randint(int(height * 0.7), int(height * 1.0))

            # ROI position
            x = np.random.randint(min_x, width - roi_width - 1 + min_x)
            y = np.random.randint(min_y, height - roi_height - 1 + min_y)

            # Extract ROI from processed image
            roi = self.processed_image[y:y+roi_height, x:x+roi_width]
            rois.append((roi, (x, y)))

        return rois

    def _build_matrices(self, points):
        """Build matrices for circle fitting."""
        points = np.array(points, dtype=np.float64)
        x = points[:, 0]
        y = points[:, 1]
        
        n = len(points)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_x2 = np.sum(x**2)
        sum_y2 = np.sum(y**2)
        sum_xy = np.sum(x * y)
        
        A = np.array([
            [sum_x2, sum_xy, sum_x],
            [sum_xy, sum_y2, sum_y],
            [sum_x,  sum_y,  n]
        ])
        
        x2y2 = x**2 + y**2
        B = np.array([
            np.sum(x * x2y2),
            np.sum(y * x2y2),
            np.sum(x2y2)
        ])
        
        return A, B

    def _fit_circle(self, points):
        """
        Fit circle to given points.
        
        Returns:
            tuple: ((center_x, center_y), radius) or (None, None)
        """
        if len(points) < 3:
            return None, None

        A, B = self._build_matrices(points)
        try:
            X = np.linalg.solve(A, B)
            u, v, w = X
            center_x = u / 2.0
            center_y = v / 2.0
            radius = np.sqrt(center_x**2 + center_y**2 + w)
            return (int(center_x), int(center_y)), int(radius)
        except np.linalg.LinAlgError:
            return None, None

    def _calculate_score(self, center, radius, points):
        """
        Calculate fitting score based on point distances to the circle.
        
        Returns:
            float: Score between 0 and 1
        """
        if center is None or points is None or len(points) == 0:
            return 0.0

        inlier_threshold = radius * 0.05  # 5% radius tolerance
        inlier_count = 0
        cx, cy = center

        for (x, y) in points:
            distance = abs(np.sqrt((x - cx)**2 + (y - cy)**2) - radius)
            if distance < inlier_threshold:
                inlier_count += 1

        return inlier_count / len(points)

    def detect_circle(self):
        """
        Detect circle using ROI-based approach and scoring.
        
        Returns:
            tuple: ((center_x, center_y), radius) or None
        """
        if self.processed_image is None:
            raise ValueError("Image not preprocessed")

        # Get points from binary image
        points = np.column_stack(np.where(self.processed_image > 0))
        if points.size == 0:
            return None

        # Convert (y,x) to (x,y)
        points = points[:, [1, 0]]

        # Generate ROIs
        rois = self._get_random_rois(points)
        
        # Process each ROI
        best_score = -1
        best_result = None
        best_roi_idx = -1

        for i, (roi_img, roi_pos) in enumerate(rois):
            # Get points in ROI
            roi_points = np.column_stack(np.where(roi_img > 0))
            if roi_points.size > 0:
                roi_points = roi_points[:, [1, 0]]  # Convert (y,x) to (x,y)
                
                # Fit circle
                center, radius = self._fit_circle(roi_points)
                if center is not None and radius is not None:
                    # Convert to global coordinates
                    global_center = (center[0] + roi_pos[0], center[1] + roi_pos[1])
                    
                    # Calculate score
                    score = self._calculate_score(global_center, radius, points)
                    
                    if score > best_score and score >= self.min_score:
                        best_score = score
                        best_result = (global_center, radius)
                        best_roi_idx = i

        if best_result is not None:
            self.result = best_result
            return best_result
        return None

    def draw_result(self, draw_all_rois=False):
        """
        Draw the detected circle on the original image.
        
        Args:
            draw_all_rois (bool): Whether to draw all ROI boundaries
            
        Returns:
            numpy.ndarray: Result image
        """
        if self.image is None or self.result is None:
            raise ValueError("No image or no circle detected")

        result_image = self.image.copy()
        center, radius = self.result
        
        # Draw the circle
        cv2.circle(result_image, center, radius, (0, 255, 0), 2)
        # Draw the center point
        cv2.circle(result_image, center, 5, (0, 0, 255), -1)

        return result_image

    def save_result(self, output_path, draw_all_rois=False):
        """
        Save the result image with detected circle.
        
        Args:
            output_path (str): Path to save the result image
            draw_all_rois (bool): Whether to draw all ROI boundaries
        """
        if self.result is None:
            raise ValueError("No circle detected")
        
        result_image = self.draw_result(draw_all_rois)
        cv2.imwrite(output_path, result_image)
        return self 