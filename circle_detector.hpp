#ifndef CIRCLE_DETECTOR_HPP
#define CIRCLE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <stdexcept>

class CircleDetector {
public:
    CircleDetector(int num_rois = 6, double min_score = 0.7);
    
    // 加载和预处理图像
    CircleDetector& loadImage(const std::string& image_path);
    CircleDetector& loadImage(const cv::Mat& image);
    CircleDetector& preprocessImage();
    
    // 主要的检测函数
    std::pair<cv::Point, int> detectCircle();
    
    // 结果可视化
    cv::Mat drawResult(bool draw_all_rois = false) const;
    CircleDetector& saveResult(const std::string& output_path, bool draw_all_rois = false);
    
    // Getter/Setter
    void setProcessedImage(const cv::Mat& processed_image) { processed_image_ = processed_image; }
    const cv::Mat& getProcessedImage() const { return processed_image_; }

private:
    // 内部辅助函数
    std::vector<std::pair<cv::Mat, cv::Point>> getRandomRois(const std::vector<cv::Point>& points);
    std::pair<cv::Mat, cv::Mat> buildMatrices(const std::vector<cv::Point>& points) const;
    std::pair<cv::Point, int> fitCircle(const std::vector<cv::Point>& points) const;
    double calculateScore(const cv::Point& center, int radius, const std::vector<cv::Point>& points) const;

    // 成员变量
    cv::Mat image_;
    cv::Mat processed_image_;
    std::pair<cv::Point, int> result_;  // (center, radius)
    int num_rois_;
    double min_score_;
};

#endif // CIRCLE_DETECTOR_HPP 