import cv2
import numpy as np
from circle_detector import CircleDetector
import time

class HoopDetector:
    def __init__(self, kernel_size=5, solidity_threshold=0.6, min_score=0.7):
        """初始化篮筐检测器
        
        Args:
            kernel_size: 形态学操作的核大小
            solidity_threshold: 密实度阈值，用于区分篮筐和方块
            min_score: 圆检测的最小分数阈值
        """
        self.kernel_size = kernel_size
        self.solidity_threshold = solidity_threshold
        self.min_score = min_score
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 创建CircleDetector实例
        self.circle_detector = CircleDetector(
            num_rois=6,
            min_score=min_score
        )
        
        # 存储处理过程中的图像
        self.original_image = None
        self.initial_binary = None
        self.binary_image = None
        self.eroded_image = None
        self.contour_image = None
        self.filtered_image = None
        self.final_binary = None
        self.result_image = None
        
        # 存储各步骤的运行时间
        self.timing = {}
        
    def reset_timing(self):
        """重置时间统计"""
        self.timing = {}
        
    def load_image(self, image_path):
        """加载图像"""
        start_time = time.time()
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        self.circle_detector.load_image(image_path)
        self.timing['load_image'] = time.time() - start_time
        return self
        
    def create_binary_image(self):
        """创建二值图像"""
        if self.original_image is None:
            raise ValueError("未加载图像")
        
        start_time = time.time()
        
        # 转换到HSV颜色空间并增强饱和度
        hsv_frame = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_frame)
        s = np.clip(s * 1.5, 0, 255).astype(np.uint8)
        hsv_enhanced = cv2.merge([h, s, v])
        
        # 创建橙色掩码（初始二值图）
        lower_bound = np.array([5, 220, 0])
        upper_bound = np.array([11, 255, 255])
        self.initial_binary = cv2.inRange(hsv_enhanced, lower_bound, upper_bound)
        
        # 清理掩码
        kernel = np.ones((5, 5), np.uint8)
        # 开运算（先腐蚀后膨胀），去除小的噪点
        mask = cv2.morphologyEx(self.initial_binary, cv2.MORPH_OPEN, kernel)
        # 闭运算（先膨胀后腐蚀），填充小孔
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # 中值滤波，去除椒盐噪声
        self.binary_image = cv2.medianBlur(mask, 23)
        
        self.timing['create_binary'] = time.time() - start_time
        return self
        
    def process_image(self):
        """处理图像，分离篮筐和干扰物"""
        if self.binary_image is None:
            raise ValueError("未创建二值图像")
        
        start_time = time.time()
        
        # 1. 腐蚀操作，断开连接
        self.eroded_image = cv2.erode(self.binary_image, self.kernel, iterations=3)
        
        # 2. 查找轮廓
        contours, _ = cv2.findContours(self.eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建轮廓图像（用于可视化）
        self.contour_image = cv2.cvtColor(self.eroded_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(self.contour_image, contours, -1, (0, 255, 0), 2)
        
        # 3. 基于密实度筛选轮廓
        filtered_contours = []
        for contour in contours:
            # 计算轮廓面积和凸包
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            
            # 计算密实度
            if hull_area > 0:
                solidity = float(area) / hull_area
                if solidity < self.solidity_threshold:
                    filtered_contours.append(contour)
        
        # 4. 创建过滤后的掩码
        self.filtered_image = np.zeros_like(self.binary_image)
        cv2.drawContours(self.filtered_image, filtered_contours, -1, 255, -1)
        
        # 5. 膨胀操作，恢复目标大小
        self.final_binary = cv2.dilate(self.filtered_image, self.kernel, iterations=2)
        
        self.timing['process_image'] = time.time() - start_time
        return self
        
    def detect_circle(self):
        """使用CircleDetector检测圆"""
        if self.final_binary is None:
            raise ValueError("未完成图像处理")
        
        start_time = time.time()
        
        # 将我们处理后的二值图传给circle_detector
        self.circle_detector.processed_image = self.final_binary
        
        # 使用circle_detector进行检测
        result = self.circle_detector.detect_circle()
        
        # 在原始图像上绘制结果
        self.result_image = self.original_image.copy()
        
        if result:
            center, radius = result
            # 绘制圆和圆心
            cv2.circle(self.result_image, center, radius, (0, 255, 0), 2)
            cv2.circle(self.result_image, center, 2, (0, 0, 255), 3)
            
        self.timing['detect_circle'] = time.time() - start_time
        return result

    def print_timing_stats(self):
        """打印时间统计信息"""
        print("\n=== Performance Statistics ===")
        total_time = sum(self.timing.values())
        
        # 打印每个步骤的时间
        for step, duration in self.timing.items():
            percentage = (duration / total_time) * 100
            print(f"{step:15s}: {duration*1000:6.2f} ms ({percentage:5.1f}%)")
        
        # 打印总时间
        print("-" * 40)
        print(f"{'Total':15s}: {total_time*1000:6.2f} ms (100.0%)")
        print(f"FPS          : {1/total_time:6.2f}")

    def show_process(self):
        """显示处理过程"""
        if self.original_image is None:
            raise ValueError("未加载图像")
            
        # 创建显示窗口
        cv2.namedWindow('Processing Steps', cv2.WINDOW_NORMAL)
        
        # 准备所有处理步骤的图像
        images = []
        titles = []
        
        # 添加原始图像
        images.append(self.original_image)
        titles.append('1. Original Image')
        
        # 添加初始二值图像
        if self.initial_binary is not None:
            initial_binary_display = cv2.cvtColor(self.initial_binary, cv2.COLOR_GRAY2BGR)
            images.append(initial_binary_display)
            titles.append('2. Initial Binary')
        
        # 添加滤波后的二值图像
        if self.binary_image is not None:
            binary_display = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR)
            images.append(binary_display)
            titles.append('3. Filtered Binary')
        
        # 添加腐蚀结果
        if self.eroded_image is not None:
            eroded_display = cv2.cvtColor(self.eroded_image, cv2.COLOR_GRAY2BGR)
            images.append(eroded_display)
            titles.append('4. Eroded Image')
        
        # 添加轮廓分析结果
        if self.contour_image is not None:
            images.append(self.contour_image)
            titles.append('5. Contour Analysis')
        
        # 添加过滤后的图像
        if self.filtered_image is not None:
            filtered_display = cv2.cvtColor(self.filtered_image, cv2.COLOR_GRAY2BGR)
            images.append(filtered_display)
            titles.append('6. Filtered Contours')
        
        # 添加最终的二值图像
        if self.final_binary is not None:
            final_display = cv2.cvtColor(self.final_binary, cv2.COLOR_GRAY2BGR)
            images.append(final_display)
            titles.append('7. Final Binary')
        
        # 添加结果图像
        if self.result_image is not None:
            images.append(self.result_image)
            titles.append('8. Detection Result')
        
        # 计算显示布局
        n_images = len(images)
        n_cols = min(4, n_images)  # 每行最多显示4张图
        n_rows = (n_images + n_cols - 1) // n_cols
        
        # 调整图像大小
        height = 300
        processed_images = []
        for img in images:
            aspect_ratio = img.shape[1] / img.shape[0]
            width = int(height * aspect_ratio)
            resized = cv2.resize(img, (width, height))
            processed_images.append(resized)
        
        # 创建画布
        max_width = max(img.shape[1] for img in processed_images)
        canvas = np.zeros((height * n_rows, max_width * n_cols, 3), dtype=np.uint8)
        
        # 放置图像和标题
        for idx, (img, title) in enumerate(zip(processed_images, titles)):
            i, j = idx // n_cols, idx % n_cols
            y_start = i * height
            x_start = j * max_width + (max_width - img.shape[1]) // 2
            
            # 放置图像
            canvas[y_start:y_start+img.shape[0], x_start:x_start+img.shape[1]] = img
            
            # 添加标题
            cv2.putText(canvas, title, (x_start + 10, y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('Processing Steps', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # 创建检测器实例
    detector = HoopDetector(
        kernel_size=5,
        solidity_threshold=0.6,
        min_score=0.7
    )
    
    try:
        # 重置时间统计
        detector.reset_timing()
        
        # 加载并处理图像
        detector.load_image("/home/niu/Desktop/wfy/pictures/hoopA.bmp")
        detector.create_binary_image()
        detector.process_image()
        result = detector.detect_circle()
        
        if result:
            center, radius = result
            print(f"检测到篮筐!")
            print(f"圆心坐标: ({center[0]}, {center[1]})")
            print(f"半径: {radius}")
        else:
            print("未检测到篮筐")
        
        # 打印时间统计
        detector.print_timing_stats()
        
        # 显示处理过程
        detector.show_process()
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main() 