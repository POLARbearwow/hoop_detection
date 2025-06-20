import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import matplotlib
    matplotlib.use('TkAgg')
except ImportError:
    print("Tkinter未找到，将使用Matplotlib的默认后端。如果图形界面有问题，请尝试安装python3-tk (或类似包)。")

class InteractiveColorSegmenter:
    def __init__(self, image_path, brush_size=5):
        self.image_path = image_path
        self.brush_size = max(1, brush_size) # 画笔最小为1x1
            
        self.img_bgr = None
        self.img_rgb = None
        self.img_hsv = None
        self.display_img_with_overlay = None # 用于显示在GUI上，带绘画笔迹的图像副本

        self.unique_hsv_pixels_all_image = np.array([])
        self.colors_for_all_unique_hsv = np.array([])

        self.fig = None
        self.ax_img = None # 子图1：原始带笔迹图像
        self.img_artist = None # Matplotlib artist for the image
        self.ax_3d_hsv = None # 子图2：3D HSV图
        self.base_hsv_scatter_artist = None
        self.painted_highlight_artist = None
        self.ax_binary_mask = None # 子图3：二值化图
        self.binary_mask_artist = None # Matplotlib artist for binary mask

        self.is_painting = False
        self.painted_hsv_values_set = set() # 存储 (h,s,v) 元组

        self._load_and_prepare_data()
        self._setup_plot()

    def _load_and_prepare_data(self):
        self.img_bgr = cv2.imread(self.image_path)
        if self.img_bgr is None:
            raise FileNotFoundError(f"错误：无法加载图像 '{self.image_path}'")

        if len(self.img_bgr.shape) == 2 or self.img_bgr.shape[2] == 1:
            self.img_bgr = cv2.cvtColor(self.img_bgr, cv2.COLOR_GRAY2BGR)
        elif self.img_bgr.shape[2] != 3:
            raise ValueError(f"图像具有不支持的通道数: {self.img_bgr.shape[2]}")

        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        self.img_hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)
        self.display_img_with_overlay = self.img_rgb.copy() # 创建副本用于叠加笔迹

        all_hsv_pixels_in_image = self.img_hsv.reshape(-1, 3)
        if all_hsv_pixels_in_image.size > 0:
            self.unique_hsv_pixels_all_image = np.unique(all_hsv_pixels_in_image, axis=0)
        
        print(f"图像中总共有 {len(self.unique_hsv_pixels_all_image)} 个独特的HSV颜色值。")

        if len(self.unique_hsv_pixels_all_image) > 0:
            hsv_for_conversion = self.unique_hsv_pixels_all_image.reshape(-1, 1, 3).astype(np.uint8)
            rgb_for_points = cv2.cvtColor(hsv_for_conversion, cv2.COLOR_HSV2RGB)
            self.colors_for_all_unique_hsv = rgb_for_points.reshape(-1, 3) / 255.0

    def _setup_plot(self):
        self.fig = plt.figure(figsize=(20, 6)) # 调整画布大小以容纳三个子图

        # 子图1: 原始图像 (带绘画笔迹)
        self.ax_img = self.fig.add_subplot(1, 3, 1)
        self.ax_img.set_title(f'原始图像 (画笔: {self.brush_size}px, 按\'c\'清除)')
        if self.display_img_with_overlay is not None:
            self.img_artist = self.ax_img.imshow(self.display_img_with_overlay)
        else: # 备用，如果display_img_with_overlay未初始化
            self.img_artist = self.ax_img.imshow(np.zeros((100,100,3),dtype=np.uint8))
        self.ax_img.axis('off')

        # 子图2: 3D HSV 颜色空间散点图
        self.ax_3d_hsv = self.fig.add_subplot(1, 3, 2, projection='3d')
        self.ax_3d_hsv.set_title('HSV颜色空间散点图')
        self.ax_3d_hsv.set_xlabel('Hue (H)'); self.ax_3d_hsv.set_ylabel('Saturation (S)'); self.ax_3d_hsv.set_zlabel('Value (V)')
        self.ax_3d_hsv.set_xlim([0, 180]); self.ax_3d_hsv.set_ylim([0, 256]); self.ax_3d_hsv.set_zlim([0, 256])

        if len(self.unique_hsv_pixels_all_image) > 0:
            self.base_hsv_scatter_artist = self.ax_3d_hsv.scatter(
                self.unique_hsv_pixels_all_image[:, 0], self.unique_hsv_pixels_all_image[:, 1], self.unique_hsv_pixels_all_image[:, 2],
                c=self.colors_for_all_unique_hsv, s=15, alpha=0.5
            )
        self.painted_highlight_artist = self.ax_3d_hsv.scatter(
            [], [], [], s=70, color='lime', edgecolor='black', marker='X', depthshade=False
        )

        # 子图3: 二值化图像
        self.ax_binary_mask = self.fig.add_subplot(1, 3, 3)
        self.ax_binary_mask.set_title('二值化图像 (基于选择的HSV范围)')
        self.ax_binary_mask.axis('off')
        # 初始化一个黑色的二值图
        init_mask_shape = (self.img_hsv.shape[0], self.img_hsv.shape[1]) if self.img_hsv is not None else (100,100)
        self.binary_mask_artist = self.ax_binary_mask.imshow(np.zeros(init_mask_shape, dtype=np.uint8), cmap='gray', vmin=0, vmax=255)

        # 连接事件处理器
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.tight_layout(pad=2.0)


    def _paint_stroke_on_image(self, x_coords, y_coords, color=[255,100,100]): # 淡红色作为笔迹颜色
        """在display_img_with_overlay上绘制笔迹"""
        if self.display_img_with_overlay is None or not x_coords or not y_coords:
            return
        
        # 确保坐标在界内并转换为整数索引
        x_indices = np.clip(np.round(x_coords).astype(int), 0, self.display_img_with_overlay.shape[1]-1)
        y_indices = np.clip(np.round(y_coords).astype(int), 0, self.display_img_with_overlay.shape[0]-1)
        
        # 使用高级索引直接修改颜色，避免循环
        # 注意：这里会直接覆盖，如果需要混合效果会更复杂
        self.display_img_with_overlay[y_indices, x_indices] = color

        if self.img_artist:
            self.img_artist.set_data(self.display_img_with_overlay)
        # self.fig.canvas.draw_idle() # 在on_mouse_motion中调用，以实时显示

    def on_mouse_press(self, event):
        if event.inaxes == self.ax_img and event.button == 1:
            self.is_painting = True
            # self.painted_hsv_values_set.clear() # 若希望每次绘画都重新开始选择，取消此行注释
            # self._reset_display_image_overlay() # 相应地，也需要重置笔迹叠加
            self._apply_paint(event)

    def on_mouse_motion(self, event):
        if self.is_painting and event.inaxes == self.ax_img:
            self._apply_paint(event)
            self.fig.canvas.draw_idle() # 实时更新笔迹显示

    def on_mouse_release(self, event):
        if event.button == 1 and self.is_painting:
            self.is_painting = False
            self._update_3d_highlights_painted()
            self._update_binary_mask_plot()
            self.fig.canvas.draw_idle() # 确保最终状态被绘制
            print(f"绘画结束。当前共选中 {len(self.painted_hsv_values_set)} 个独特的HSV值。")

    def _apply_paint(self, event): # 统一处理绘画逻辑
        if self.img_hsv is None or self.display_img_with_overlay is None: return
        
        x_center, y_center = event.xdata, event.ydata
        if x_center is None or y_center is None: return # 点击到子图外

        half_b = self.brush_size // 2
        
        # 生成画笔覆盖的像素坐标网格
        y_coords_brush, x_coords_brush = np.ogrid[
            max(0, int(round(y_center)) - half_b) : min(self.img_hsv.shape[0], int(round(y_center)) + half_b + (self.brush_size % 2)),
            max(0, int(round(x_center)) - half_b) : min(self.img_hsv.shape[1], int(round(x_center)) + half_b + (self.brush_size % 2))
        ]
        
        if x_coords_brush.size == 0 or y_coords_brush.size == 0: return

        # 1. 收集HSV值
        brush_area_hsv = self.img_hsv[y_coords_brush, x_coords_brush]
        for hsv_pixel in brush_area_hsv.reshape(-1, 3):
            self.painted_hsv_values_set.add(tuple(hsv_pixel))

        # 2. 在display_img_with_overlay上绘制笔迹 (使用原始坐标，而不是网格的广播坐标)
        # _paint_stroke_on_image 需要的是一个扁平的x, y坐标列表或数组
        # 为了简化，我们直接使用画笔区域的边界来标记，而不是精确的圆形/方形笔刷的每个点
        # 这里我们用一个简化的方法，直接给整个矩形区域上色
        y_min_idx, y_max_idx = y_coords_brush.min(), y_coords_brush.max() + 1
        x_min_idx, x_max_idx = x_coords_brush.min(), x_coords_brush.max() + 1
        
        if y_min_idx < y_max_idx and x_min_idx < x_max_idx : # 确保切片有效
            self.display_img_with_overlay[y_min_idx:y_max_idx, x_min_idx:x_max_idx] = [255, 100, 100] # 淡红色笔迹

        if self.img_artist:
             self.img_artist.set_data(self.display_img_with_overlay)
        # self.fig.canvas.draw_idle() 将在 on_mouse_motion 中调用


    def _update_3d_highlights_painted(self):
        if not self.painted_highlight_artist: return
        if not self.painted_hsv_values_set:
            self.painted_highlight_artist._offsets3d = ([], [], [])
        else:
            h_coords = [p[0] for p in self.painted_hsv_values_set]
            s_coords = [p[1] for p in self.painted_hsv_values_set]
            v_coords = [p[2] for p in self.painted_hsv_values_set]
            self.painted_highlight_artist._offsets3d = (h_coords, s_coords, v_coords)
        # self.fig.canvas.draw_idle() # 由调用者负责刷新

    def _update_binary_mask_plot(self):
        if not self.binary_mask_artist or self.img_hsv is None: return

        if not self.painted_hsv_values_set:
            empty_mask = np.zeros((self.img_hsv.shape[0], self.img_hsv.shape[1]), dtype=np.uint8)
            self.binary_mask_artist.set_data(empty_mask)
            print("没有选中的HSV值，二值化图已清空。")
            # self.fig.canvas.draw_idle() # 由调用者负责刷新
            return

        h_v = np.array([p[0] for p in self.painted_hsv_values_set])
        s_v = np.array([p[1] for p in self.painted_hsv_values_set])
        v_v = np.array([p[2] for p in self.painted_hsv_values_set])

        h_min, h_max = np.min(h_v), np.max(h_v)
        s_min, s_max = np.min(s_v), np.max(s_v)
        v_min, v_max = np.min(v_v), np.max(v_v)
        
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        print(f"\n计算得到的HSV范围用于二值化:")
        print(f"  H: {h_min}-{h_max}, S: {s_min}-{s_max}, V: {v_min}-{v_max}")
        if h_max - h_min > 120 : # 一个简单的启发式检测，提示Hue范围过大可能存在的问题
            print("  注意: 选择的色调(H)范围非常宽。如果选择的颜色跨越了红色(0/179)的边界，")
            print("        当前的简单范围计算可能包含非期望颜色，或遗漏部分期望颜色。")
            print("        例如，同时选择色调为5和色调为175的红色，计算出的H范围[5-175]会包含蓝绿黄等。")
            print("        这种情况下，建议分多次选择，或更细致地绘画。")


        binary_mask = cv2.inRange(self.img_hsv, lower_bound, upper_bound)
        self.binary_mask_artist.set_data(binary_mask)
        # self.fig.canvas.draw_idle() # 由调用者负责刷新
        
    def _reset_display_image_overlay(self):
        if self.img_rgb is not None:
            self.display_img_with_overlay = self.img_rgb.copy()
            if self.img_artist:
                self.img_artist.set_data(self.display_img_with_overlay)

    def on_key_press(self, event):
        if event.key == 'c':
            self.painted_hsv_values_set.clear()
            self._reset_display_image_overlay() # 重置图像上的笔迹
            self._update_3d_highlights_painted() # 清除3D图高亮
            self._update_binary_mask_plot()      # 清除二值化图
            self.fig.canvas.draw_idle()          # 刷新显示
            print("所有选择、笔迹和二值化图已清除。")

    def show(self):
        if self.img_bgr is None: return
        plt.show()

if __name__ == '__main__':
    image_file = '/opt/MVS/bin/Temp/Data/Image_20250611164813887.bmp' # <--- 修改这里!!
    brush_s = 7 

    try:
        if image_file == 'your_image_path.jpg':
            try:
                with open(image_file, 'rb') as f_check: pass
            except FileNotFoundError:
                raise FileNotFoundError("使用演示图片")

        print(f"尝试加载用户指定图片: {image_file}")
        segmenter = InteractiveColorSegmenter(image_file, brush_size=brush_s)
        segmenter.show()

    except FileNotFoundError:
        print(f"图片 '{image_file}' 未找到或未指定。正在创建并使用演示图片...")
        # (创建演示图片的代码与之前类似，这里省略以保持简洁，你可以复用之前的)
        demo_img_bgr = np.zeros((150, 200, 3), dtype=np.uint8)
        demo_img_bgr[10:70, 10:70] = [255, 100, 50]; demo_img_bgr[10:70, 80:140] = [50, 255, 100]
        demo_img_bgr[80:140, 10:70] = [100, 50, 255]; demo_img_bgr[80:140, 80:140] = [200, 200, 50]
        demo_img_bgr[0:150, 150:200] = [70, 70, 70]
        cv2.imwrite('demo_segmenter_image.png', demo_img_bgr)
        print("演示图片 'demo_segmenter_image.png' 已创建。")
        segmenter = InteractiveColorSegmenter('demo_segmenter_image.png', brush_size=brush_s)
        segmenter.show()
        
    except Exception as e_main:
        print(f"发生了一个未预料的错误: {e_main}")
        import traceback
        traceback.print_exc()