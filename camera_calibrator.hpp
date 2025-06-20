#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class CameraCalibrator {
public:
    CameraCalibrator() = default;
    
    // 从图片目录进行标定
    bool calibrateFromImages(const std::string& image_dir, 
                           int board_width, 
                           int board_height, 
                           float square_size);
    
    // 验证标定结果
    bool verifyCalibration(const std::string& calib_file, 
                          const std::string& test_image);
    
    // 保存标定结果
    bool saveCalibrationResult(const std::string& filename);
    
    // 获取标定结果
    cv::Mat getCameraMatrix() const { return camera_matrix_; }
    cv::Mat getDistCoeffs() const { return dist_coeffs_; }
    
private:
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    cv::Size image_size_;
    double rms_error_ = 0.0;
    
    // 辅助函数：检测单张图片中的棋盘格角点
    bool detectChessboardCorners(const cv::Mat& image,
                               const cv::Size& board_size,
                               std::vector<cv::Point2f>& corners);
}; 