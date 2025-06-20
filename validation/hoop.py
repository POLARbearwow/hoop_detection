import cv2
import numpy as np

def process_and_compare_methods(image_path):
    """
    加载图像，同时使用轮廓拟合和霍夫变换检测篮球筐，并在网格中对比结果。
    """
    # --- 0. 设置缩放比例 ---
    scale_percent = 50

    # --- 1. 加载图像 ---
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法加载图片，请检查路径: {image_path}")
        return
    print("图像加载成功，开始处理...")

    # --- 2. 预处理 (颜色分割和二值化) ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # --- 修改部分：已移除不必要的mask2 ---
    # 根据您找到的参数，我们只需要一个HSV范围。
    lower_bound = np.array([13, 0, 0])
    upper_bound = np.array([179, 190, 241])
    
    # 直接生成最终的二值化掩码
    binarized_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    binarized_mask = cv2.bitwise_not(binarized_mask)
    cv2.imshow("tt",binarized_mask)
    # print(binarized_mask)
    
    # 形态学处理，清理掩码
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_opened = cv2.morphologyEx(binarized_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask_opened = cv2.morphologyEx(mask_opened, cv2.MORPH_OPEN, kernel, iterations=2)
    print("预处理完成，生成二值化图。")

    # --- 3. 方法一：轮廓拟合 (Contour Fitting) ---
    result_contour = image.copy()
    contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # vstack所有轮廓点以处理被分割的轮廓
        all_points = np.vstack([cnt for cnt in contours])
        if len(all_points) >= 5:
            ellipse = cv2.fitEllipse(all_points)
            cv2.ellipse(result_contour, ellipse, (0, 255, 0), 5) # 绿色椭圆
            print("方法一 (轮廓拟合) 检测成功。")

    # --- 4. 方法二：霍夫圆变换 (Hough Circle Transform) ---
    result_hough = image.copy()
    circles = cv2.HoughCircles(mask_opened, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=100, param2=25, minRadius=int(image.shape[1]*0.15), maxRadius=int(image.shape[1]*0.4))
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(result_hough, (i[0], i[1]), i[2], (0, 0, 255), 5) # 红色圆形
        print(f"方法二 (霍夫变换) 检测成功，找到 {len(circles[0])} 个圆。")
    else:
        print("方法二 (霍夫变换) 未找到圆形。")

    # --- 5. 准备并拼接网格视图 ---
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    font = cv2.FONT_HERSHEY_SIMPLEX

    img_resized = cv2.resize(image, dim)
    cv2.putText(result_contour, "Contour Fit (Green)", (10, 50), font, 1.5, (0, 255, 0), 3)
    cv2.putText(result_hough, "Hough Transform (Red)", (10, 50), font, 1.5, (0, 0, 255), 3)
    contour_resized = cv2.resize(result_contour, dim)
    hough_resized = cv2.resize(result_hough, dim)
    
    # 这里的mask_opened就是标准的二值化图 (目标为白)
    binarized_resized = cv2.resize(cv2.cvtColor(mask_opened, cv2.COLOR_GRAY2BGR), dim)

    # 拼接网格
    top_row = np.hstack((img_resized, binarized_resized))
    bottom_row = np.hstack((contour_resized, hough_resized))
    combined_view = np.vstack((top_row, bottom_row))

    # --- 6. 显示最终对比图 ---
    cv2.imshow('Contour Fit vs. Hough Transform', combined_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- 主程序入口 ---
if __name__ == '__main__':
    image_filename = "/home/niu/Desktop/hoop1.jpg"
    process_and_compare_methods(image_filename)