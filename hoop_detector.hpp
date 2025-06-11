#pragma once

#include <opencv2/opencv.hpp>
#include <chrono>
#include <map>
#include <string>
#include "circle_detector.hpp"

class HoopDetector {
public:
    HoopDetector(int kernel_size, double solidity_threshold, double min_score);
    
    // 加载和处理图像
    HoopDetector& loadImage(const std::string& image_path);
    HoopDetector& loadImage(const cv::Mat& image);
    HoopDetector& createBinaryImage();
    HoopDetector& processImage();
    
    // 主要的检测函数
    std::pair<cv::Point, int> detectCircle();
    
    // 可视化和调试
    cv::Mat showProcess() const;
    void printTimingStats() const;
    
    // 获取处理后的图像
    const cv::Mat& getOriginalImage() const { return original_image_; }
    const cv::Mat& getInitialBinary() const { return initial_binary_; }
    const cv::Mat& getBinaryImage() const { return binary_image_; }
    const cv::Mat& getErodedImage() const { return eroded_image_; }
    const cv::Mat& getContourImage() const { return contour_image_; }
    const cv::Mat& getFilteredImage() const { return filtered_image_; }
    const cv::Mat& getFinalBinary() const { return final_binary_; }
    const cv::Mat& getResultImage() const { return result_image_; }
    const cv::Mat& getLargestContourImage() const { return largest_contour_image_; }
    const cv::Mat& getRemainingContoursImage() const { return remaining_contours_image_; }
    const cv::Mat& getSquareRegionImage() const { return square_region_image_; }

    void resetTiming();

private:
    // 新增：日志控制相关
    std::chrono::steady_clock::time_point last_log_time_;
    const double log_interval_ = 1.0;  // 日志输出间隔（秒）
    void logIfNeeded(const std::string& message);
    
    // 新增：圆拟合相关
    std::pair<cv::Point2f, float> fitCircle3Points(
        const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3);
    std::pair<cv::Point, int> fitCircleRANSAC(
        const std::vector<cv::Point>& points, 
        int n_iterations = 200, 
        float threshold_dist = 8.0);
    
    // 成员变量
    CircleDetector circle_detector_;
    int kernel_size_;
    double solidity_threshold_;
    cv::Mat kernel_;
    
    // 图像保存相关
    std::chrono::steady_clock::time_point last_save_time_;
    int save_counter_ = 0;
    const double save_interval_ = 2.0; // 保存间隔（秒）
    
    // 图像处理过程中的中间结果
    cv::Mat original_image_;
    cv::Mat initial_binary_;
    cv::Mat binary_image_;
    cv::Mat eroded_image_;
    cv::Mat contour_image_;
    cv::Mat filtered_image_;
    cv::Mat largest_contour_image_;    // 最大轮廓图像
    cv::Mat remaining_contours_image_; // 剩余轮廓图像
    cv::Mat square_region_image_;     // 最像正方形的区域图像
    cv::Mat final_binary_;
    cv::Mat result_image_;
    
    // 时间统计
    std::map<std::string, double> timing_;
}; 