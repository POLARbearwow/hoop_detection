import cv2
import numpy as np
import math

# --- 参数配置 ---
# 1. 橙色的 HSV 范围
LOWER_ORANGE = np.array([5, 126, 44])
UPPER_ORANGE = np.array([14, 243, 91]) # 经测试，这个上限对某些橙色更有效

# 2. 椭圆拟合的最小轮廓点数
MIN_CONTOUR_POINTS_FOR_ELLIPSE = 50

# 3. 三维坐标估算参数 (重要：这些参数需要通过相机标定获得！)
CAMERA_FOCAL_LENGTH_X = 800
CAMERA_FOCAL_LENGTH_Y = 800
CAMERA_PRINCIPAL_POINT_X = 320 # 假设图像宽度为640
CAMERA_PRINCIPAL_POINT_Y = 240 # 假设图像高度为480
REAL_HOOP_RADIUS_METERS = 0.225

def estimate_3d_coordinates(ellipse_center_px, ellipse_axes_px, real_radius_m, fx, fy, cx, cy):
    """
    根据椭圆参数和相机内参估算圆环中心的三维坐标。
    """
    u, v = ellipse_center_px
    minor_axis_len, major_axis_len = ellipse_axes_px
    a_px = major_axis_len / 2.0

    if a_px <= 0:
        return None

    f_avg = (fx + fy) / 2.0
    Z = (f_avg * real_radius_m) / a_px
    X = ((u - cx) * Z) / fx
    Y = ((v - cy) * Z) / fy

    return (X, Y, Z)

# --- 修改：函数名改为 process_image，逻辑更清晰 ---
def process_image(image_path):
    # --- 修改：从加载图片开始，而不是打开摄像头 ---
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"错误：无法加载图片，请检查路径: {image_path}")
        return

    print("图片加载成功，开始处理...")
    
    # 根据图片尺寸更新主点坐标 (如果未手动指定)
    # 这是一个好习惯，让代码更具适应性
    frame_height, frame_width = frame.shape[:2]
    global CAMERA_PRINCIPAL_POINT_X, CAMERA_PRINCIPAL_POINT_Y
    CAMERA_PRINCIPAL_POINT_X = frame_width / 2
    CAMERA_PRINCIPAL_POINT_Y = frame_height / 2

    # --- 后续的处理流程与原代码完全相同 ---

    # 1. 预处理：高斯模糊
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # 2. 颜色分割
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(hsv_frame, LOWER_ORANGE, UPPER_ORANGE)
    kernel = np.ones((5, 5), np.uint8)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)

    # 结合Canny边缘检测和形态学操作
    canny_edges = cv2.Canny(orange_mask, 100, 150)
    morph_kernel = np.ones((3, 3), np.uint8)
    canny_morph = canny_edges.copy()
    
    # 膨胀和腐蚀操作
    canny_morph = cv2.dilate(canny_morph, morph_kernel, iterations=5)
    canny_morph = cv2.erode(canny_morph, morph_kernel, iterations=9)
    
    kernel = np.ones((6, 6), np.uint8)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. 轮廓检测
    contours, _ = cv2.findContours(canny_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    total_contour_area = 0
    for contour in contours:
        valid_contours.append(contour)
        total_contour_area += cv2.contourArea(contour)
    
    detected_ellipses_info = []
    
    if len(valid_contours) > 0:
        try:
            all_points = np.vstack(valid_contours)
            
            if len(all_points) >= MIN_CONTOUR_POINTS_FOR_ELLIPSE:
                ellipse = cv2.fitEllipse(all_points)
                center_px = (int(ellipse[0][0]), int(ellipse[0][1]))
                axes_len_px = (ellipse[1][0], ellipse[1][1])
                a_semi_axis_px = axes_len_px[1] / 2.0
                b_semi_axis_px = axes_len_px[0] / 2.0
                
                aspect_ratio = a_semi_axis_px / b_semi_axis_px if b_semi_axis_px > 0 else float('inf')
                if aspect_ratio <= 5.0 and a_semi_axis_px >= 10:
                    cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
                    cv2.circle(frame, center_px, 5, (0, 0, 255), -1)

                    info_text_ellipse = f"Ellipse - Center:({center_px[0]},{center_px[1]}), a:{a_semi_axis_px:.1f}, b:{b_semi_axis_px:.1f}"
                    detected_ellipses_info.append({
                        "text_display_y_offset": 0,
                        "info_text_ellipse": info_text_ellipse,
                        "center_px": center_px,
                        "axes_len_px": axes_len_px
                    })
                    
                    stats_text = f"Valid contours: {len(valid_contours)}, Total area: {total_contour_area:.0f}px"
                    cv2.putText(frame, stats_text, (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except cv2.error as e:
            # print(f"椭圆拟合失败: {e}")
            pass

    y_offset_start = 30
    for i, info in enumerate(detected_ellipses_info):
        cv2.putText(frame, info["info_text_ellipse"], (10, y_offset_start + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 4. 估算三维坐标
        coords_3d = estimate_3d_coordinates(
            info["center_px"], info["axes_len_px"], REAL_HOOP_RADIUS_METERS,
            CAMERA_FOCAL_LENGTH_X, CAMERA_FOCAL_LENGTH_Y,
            CAMERA_PRINCIPAL_POINT_X, CAMERA_PRINCIPAL_POINT_Y
        )
        if coords_3d:
            X, Y, Z = coords_3d
            info_text_3d = f"3D Est: X={X:.2f}m, Y={Y:.2f}m, Z={Z:.2f}m"
            cv2.putText(frame, info_text_3d, (10, y_offset_start + 20 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "3D Est: N/A", (10, y_offset_start + 20 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- 修改：显示所有结果窗口 ---
    cv2.imshow("Processed Image", frame)
    cv2.imshow("Orange Mask", orange_mask)
    cv2.imshow("Canny + Morphology", canny_morph)

    # --- 修改：等待用户按键，然后关闭所有窗口 ---
    print("处理完成。按任意键关闭所有窗口。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- 修改：在这里设置您要处理的图片路径 ---
    IMAGE_PATH = "/opt/MVS/bin/Temp/Data/hoopsC.bmp" 
    process_image(IMAGE_PATH)