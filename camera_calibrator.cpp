#include "camera_calibrator.hpp"
#include <iostream>
#include <filesystem>

bool CameraCalibrator::calibrateFromImages(const std::string& image_dir, 
                                         int board_width, 
                                         int board_height, 
                                         float square_size) {
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<cv::String> image_files;
    cv::Size board_size(board_width, board_height);

    // 生成标准棋盘格角点的三维坐标
    std::vector<cv::Point3f> obj;
    for (int i = 0; i < board_height; i++) {
        for (int j = 0; j < board_width; j++) {
            obj.push_back(cv::Point3f(j * square_size, i * square_size, 0));
        }
    }

    // 获取目录下的所有图片
    std::string pattern = image_dir + "/*.jpg";
    cv::glob(pattern, image_files);
    if (image_files.empty()) {
        pattern = image_dir + "/*.png";
        cv::glob(pattern, image_files);
    }

    if (image_files.empty()) {
        std::cerr << "[CameraCalibrator] 错误: 在目录 " << image_dir << " 中未找到图片文件" << std::endl;
        return false;
    }

    std::cout << "[CameraCalibrator] 找到 " << image_files.size() << " 张图片" << std::endl;

    // 创建窗口用于显示
    cv::namedWindow("Calibration", cv::WINDOW_NORMAL);

    // 处理每张图片
    for (size_t i = 0; i < image_files.size(); i++) {
        cv::Mat img = cv::imread(image_files[i]);
        if (img.empty()) {
            std::cout << "[CameraCalibrator] 无法读取图片: " << image_files[i] << std::endl;
            continue;
        }

        if (image_size_.empty()) {
            image_size_ = img.size();
        }

        std::vector<cv::Point2f> corners;
        bool found = detectChessboardCorners(img, board_size, corners);

        if (found) {
            image_points.push_back(corners);
            object_points.push_back(obj);

            // 显示处理结果
            cv::imshow("Calibration", img);
            cv::waitKey(500); // 显示500ms
        }

        std::cout << "[CameraCalibrator] 处理图片 " << (i + 1) << "/" << image_files.size() 
                  << ": " << (found ? "成功" : "失败") << std::endl;
    }

    cv::destroyWindow("Calibration");

    if (image_points.size() < 3) {
        std::cerr << "[CameraCalibrator] 错误: 可用的图片数量不足，至少需要3张有效的标定图片" << std::endl;
        return false;
    }

    std::cout << "\n[CameraCalibrator] 开始标定..." << std::endl;

    camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
    dist_coeffs_ = cv::Mat::zeros(8, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs;

    // 执行标定
    rms_error_ = cv::calibrateCamera(object_points, image_points, image_size_,
                                   camera_matrix_, dist_coeffs_, rvecs, tvecs);

    std::cout << "\n=== 标定结果 ===" << std::endl;
    std::cout << "重投影误差: " << rms_error_ << std::endl;
    std::cout << "\n相机内参矩阵:\n" << camera_matrix_ << std::endl;
    std::cout << "\n畸变系数:\n" << dist_coeffs_ << std::endl;

    return true;
}

bool CameraCalibrator::detectChessboardCorners(const cv::Mat& image,
                                             const cv::Size& board_size,
                                             std::vector<cv::Point2f>& corners) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    bool found = cv::findChessboardCorners(gray, board_size, corners);

    if (found) {
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

        // 绘制角点
        cv::drawChessboardCorners(image, board_size, corners, found);
    }

    return found;
}

bool CameraCalibrator::saveCalibrationResult(const std::string& filename) {
    try {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            std::cerr << "[CameraCalibrator] 无法创建标定结果文件: " << filename << std::endl;
            return false;
        }

        time_t now = time(nullptr);
        fs << "calibration_date" << asctime(localtime(&now));
        fs << "image_width" << image_size_.width;
        fs << "image_height" << image_size_.height;
        fs << "camera_matrix" << camera_matrix_;
        fs << "distortion_coefficients" << dist_coeffs_;
        fs << "rms_error" << rms_error_;

        fs.release();
        std::cout << "[CameraCalibrator] 标定结果已保存到: " << filename << std::endl;
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "[CameraCalibrator] 保存标定结果时发生错误: " << e.what() << std::endl;
        return false;
    }
}

bool CameraCalibrator::verifyCalibration(const std::string& calib_file, 
                                       const std::string& test_image) {
    // 读取标定参数
    cv::FileStorage fs(calib_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[CameraCalibrator] 无法打开标定文件: " << calib_file << std::endl;
        return false;
    }

    fs["camera_matrix"] >> camera_matrix_;
    fs["distortion_coefficients"] >> dist_coeffs_;
    fs.release();

    // 读取测试图片
    cv::Mat img = cv::imread(test_image);
    if (img.empty()) {
        std::cerr << "[CameraCalibrator] 无法读取测试图片: " << test_image << std::endl;
        return false;
    }

    // 创建窗口
    cv::namedWindow("Original", cv::WINDOW_NORMAL);
    cv::namedWindow("Undistorted", cv::WINDOW_NORMAL);

    // 执行畸变校正
    cv::Mat undistorted;
    cv::undistort(img, undistorted, camera_matrix_, dist_coeffs_);

    // 显示结果
    cv::imshow("Original", img);
    cv::imshow("Undistorted", undistorted);
    
    std::cout << "[CameraCalibrator] 按任意键退出..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();

    return true;
} 