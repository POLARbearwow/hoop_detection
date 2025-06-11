#include "circle_detector.hpp"
#include <random>
#include <algorithm>
#include <iostream>

CircleDetector::CircleDetector(int num_rois, double min_score)
    : num_rois_(num_rois), min_score_(min_score) {
    std::cout << "[CircleDetector] 构造函数被调用" << std::endl;
    std::cout << "[CircleDetector] 参数: num_rois=" << num_rois 
              << ", min_score=" << min_score << std::endl;
}

CircleDetector& CircleDetector::loadImage(const std::string& image_path) {
    std::cout << "[CircleDetector] 从文件加载图像: " << image_path << std::endl;
    image_ = cv::imread(image_path);
    if (image_.empty()) {
        throw std::runtime_error("[CircleDetector] Failed to load image from " + image_path);
    }
    return *this;
}

CircleDetector& CircleDetector::loadImage(const cv::Mat& image) {
    std::cout << "[CircleDetector] 从Mat加载图像" << std::endl;
    
    if (image.empty()) {
        throw std::runtime_error("[CircleDetector] Input image is empty");
    }
    
    std::cout << "[CircleDetector] 图像信息: " << image.size() 
              << " 类型: " << image.type() 
              << " 通道数: " << image.channels() << std::endl;
    
    // 确保图像是BGR格式
    if (image.type() != CV_8UC3) {
        std::cout << "[CircleDetector] 警告：输入图像不是BGR格式，尝试转换..." << std::endl;
        if (image.channels() == 3) {
            cv::Mat temp;
            image.convertTo(temp, CV_8UC3);
            image_ = temp.clone();
        } else {
            throw std::runtime_error("[CircleDetector] 不支持的图像格式");
        }
    } else {
        image_ = image.clone();
    }
    
    return *this;
}

CircleDetector& CircleDetector::preprocessImage() {
    if (image_.empty()) {
        throw std::runtime_error("[CircleDetector] No image loaded");
    }

    try {
        std::cout << "[CircleDetector] 开始预处理图像..." << std::endl;
        
        // 转换到HSV并增强饱和度
        cv::Mat hsv_frame;
        std::cout << "[CircleDetector] 转换到HSV空间..." << std::endl;
        cv::cvtColor(image_, hsv_frame, cv::COLOR_BGR2HSV);
        
        std::vector<cv::Mat> hsv_channels;
        cv::split(hsv_frame, hsv_channels);
        
        // 增强饱和度
        std::cout << "[CircleDetector] 增强饱和度..." << std::endl;
        cv::multiply(hsv_channels[1], 1.5, hsv_channels[1]);
        cv::merge(hsv_channels, hsv_frame);

        // 创建颜色掩码
        std::cout << "[CircleDetector] 创建颜色掩码..." << std::endl;
        cv::Scalar lower_bound(5, 220, 0);
        cv::Scalar upper_bound(11, 255, 255);
        cv::Mat mask;
        cv::inRange(hsv_frame, lower_bound, upper_bound, mask);

        // 清理掩码
        std::cout << "[CircleDetector] 清理掩码..." << std::endl;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        cv::medianBlur(mask, processed_image_, 23);
        
        std::cout << "[CircleDetector] 预处理完成" << std::endl;
        
    } catch (const cv::Exception& e) {
        std::cerr << "[CircleDetector] OpenCV错误: " << e.what() << std::endl;
        throw;
    }

    return *this;
}

std::vector<std::pair<cv::Mat, cv::Point>> CircleDetector::getRandomRois(
    const std::vector<cv::Point>& points) {
    
    std::cout << "[CircleDetector] 生成随机ROI..." << std::endl;
    
    if (points.empty()) {
        std::cerr << "[CircleDetector] 警告: 没有找到任何点" << std::endl;
        return {};
    }
    
    // 找到点的边界
    cv::Rect bounds = cv::boundingRect(points);
    int width = bounds.width;
    int height = bounds.height;
    
    std::cout << "[CircleDetector] 点集边界: " << bounds << std::endl;

    // 检查边界是否有效
    if (width <= 0 || height <= 0) {
        std::cerr << "[CircleDetector] 错误: 无效的边界大小" << std::endl;
        return {};
    }

    // 确保边界在图像范围内
    if (bounds.x < 0 || bounds.y < 0 || 
        bounds.x + bounds.width > processed_image_.cols || 
        bounds.y + bounds.height > processed_image_.rows) {
        std::cerr << "[CircleDetector] 错误: 边界超出图像范围" << std::endl;
        return {};
    }

    std::vector<std::pair<cv::Mat, cv::Point>> rois;
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < num_rois_; ++i) {
        try {
            // ROI大小 (70%-100%的点分布范围)
            int min_width = std::max(int(width * 0.7), 5);  // 最小宽度为5像素
            int max_width = std::min(width, processed_image_.cols - bounds.x);
            int min_height = std::max(int(height * 0.7), 5);  // 最小高度为5像素
            int max_height = std::min(height, processed_image_.rows - bounds.y);
            
            // 确保ROI大小有效
            if (min_width >= max_width || min_height >= max_height) {
                std::cerr << "[CircleDetector] 警告: ROI " << i << " 大小无效" << std::endl;
                continue;
            }
            
            std::uniform_int_distribution<> w_dist(min_width, max_width);
            std::uniform_int_distribution<> h_dist(min_height, max_height);
            int roi_width = w_dist(gen);
            int roi_height = h_dist(gen);

            // ROI位置
            int max_x = std::min(processed_image_.cols - roi_width, bounds.x + width - 1);
            int max_y = std::min(processed_image_.rows - roi_height, bounds.y + height - 1);
            
            // 确保位置范围有效
            if (bounds.x > max_x || bounds.y > max_y) {
                std::cerr << "[CircleDetector] 警告: ROI " << i << " 位置范围无效" << std::endl;
                continue;
            }
            
            std::uniform_int_distribution<> x_dist(bounds.x, max_x);
            std::uniform_int_distribution<> y_dist(bounds.y, max_y);
            int x = x_dist(gen);
            int y = y_dist(gen);

            // 最后一次检查ROI是否在图像范围内
            cv::Rect roi_rect(x, y, roi_width, roi_height);
            if (roi_rect.x < 0 || roi_rect.y < 0 || 
                roi_rect.x + roi_rect.width > processed_image_.cols || 
                roi_rect.y + roi_rect.height > processed_image_.rows) {
                std::cerr << "[CircleDetector] 警告: ROI " << i << " 超出图像范围" << std::endl;
                continue;
            }
            
            std::cout << "[CircleDetector] ROI " << i << ": " << roi_rect << std::endl;
            rois.emplace_back(processed_image_(roi_rect), cv::Point(x, y));
            
        } catch (const cv::Exception& e) {
            std::cerr << "[CircleDetector] ROI " << i << " 生成错误: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[CircleDetector] ROI " << i << " 未知错误: " << e.what() << std::endl;
        }
    }

    std::cout << "[CircleDetector] 成功生成 " << rois.size() << " 个ROI" << std::endl;
    return rois;
}

std::pair<cv::Mat, cv::Mat> CircleDetector::buildMatrices(
    const std::vector<cv::Point>& points) const {
    
    if (points.size() < 3) {
        throw std::runtime_error("[CircleDetector] 点数不足以拟合圆");
    }
    
    cv::Mat_<double> A(3, 3, 0.0);
    cv::Mat_<double> B(3, 1, 0.0);
    
    try {
        int n = points.size();
        double sum_x = 0, sum_y = 0;
        double sum_x2 = 0, sum_y2 = 0;
        double sum_xy = 0;
        double sum_x2y2 = 0;
        double sum_xx2y2 = 0;
        double sum_yx2y2 = 0;

        // 计算所需的和
        for (const auto& point : points) {
            double x = point.x;
            double y = point.y;
            double x2 = x * x;
            double y2 = y * y;
            double xy = x * y;
            double x2y2 = x2 + y2;

            sum_x += x;
            sum_y += y;
            sum_x2 += x2;
            sum_y2 += y2;
            sum_xy += xy;
            sum_x2y2 += x2y2;
            sum_xx2y2 += x * x2y2;
            sum_yx2y2 += y * x2y2;
        }

        // 构建矩阵A
        A(0,0) = sum_x2;    A(0,1) = sum_xy;     A(0,2) = sum_x;
        A(1,0) = sum_xy;    A(1,1) = sum_y2;     A(1,2) = sum_y;
        A(2,0) = sum_x;     A(2,1) = sum_y;      A(2,2) = n;

        // 构建矩阵B
        B(0,0) = sum_xx2y2;
        B(1,0) = sum_yx2y2;
        B(2,0) = sum_x2y2;
        
    } catch (const std::exception& e) {
        std::cerr << "[CircleDetector] 矩阵构建错误: " << e.what() << std::endl;
        throw;
    }

    return {A, B};
}

std::pair<cv::Point, int> CircleDetector::fitCircle(
    const std::vector<cv::Point>& points) const {
    
    if (points.size() < 3) {
        std::cerr << "[CircleDetector] 点数不足以拟合圆" << std::endl;
        return {cv::Point(), 0};
    }

    try {
        auto [A, B] = buildMatrices(points);
        cv::Mat X;

        if (!cv::solve(A, B, X)) {
            std::cerr << "[CircleDetector] 无法求解圆方程" << std::endl;
            return {cv::Point(), 0};
        }

        double u = X.at<double>(0);
        double v = X.at<double>(1);
        double w = X.at<double>(2);

        int center_x = static_cast<int>(u / 2.0);
        int center_y = static_cast<int>(v / 2.0);
        int radius = static_cast<int>(std::sqrt(center_x*center_x + center_y*center_y + w));
        
        std::cout << "[CircleDetector] 拟合圆: 中心=(" << center_x << "," << center_y 
                  << ") 半径=" << radius << std::endl;

        return {cv::Point(center_x, center_y), radius};
        
    } catch (const std::exception& e) {
        std::cerr << "[CircleDetector] 圆拟合错误: " << e.what() << std::endl;
        return {cv::Point(), 0};
    }
}

double CircleDetector::calculateScore(
    const cv::Point& center, 
    int radius, 
    const std::vector<cv::Point>& points) const {
    
    if (points.empty() || radius <= 0) {
        return 0.0;
    }

    try {
        double inlier_threshold = radius * 0.05;  // 5% radius tolerance
        int inlier_count = 0;

        for (const auto& point : points) {
            double dx = point.x - center.x;
            double dy = point.y - center.y;
            double distance = std::abs(std::sqrt(dx*dx + dy*dy) - radius);
            if (distance < inlier_threshold) {
                ++inlier_count;
            }
        }

        double score = static_cast<double>(inlier_count) / points.size();
        std::cout << "[CircleDetector] 圆拟合得分: " << score 
                  << " (内点: " << inlier_count << "/" << points.size() << ")" << std::endl;
        
        return score;
        
    } catch (const std::exception& e) {
        std::cerr << "[CircleDetector] 计算得分错误: " << e.what() << std::endl;
        return 0.0;
    }
}

std::pair<cv::Point, int> CircleDetector::detectCircle() {
    if (processed_image_.empty()) {
        throw std::runtime_error("[CircleDetector] Image not preprocessed");
    }

    try {
        std::cout << "[CircleDetector] 开始检测圆..." << std::endl;
        
        // 获取二值图像中的点
        std::vector<cv::Point> points;
        cv::findNonZero(processed_image_, points);
        
        std::cout << "[CircleDetector] 找到 " << points.size() << " 个非零点" << std::endl;
        
        if (points.empty()) {
            std::cout << "[CircleDetector] 未找到任何点，返回空结果" << std::endl;
            return {cv::Point(), 0};
        }

        // 生成ROIs
        auto rois = getRandomRois(points);
        
        if (rois.empty()) {
            std::cout << "[CircleDetector] 未生成任何ROI，返回空结果" << std::endl;
            return {cv::Point(), 0};
        }
        
        // 处理每个ROI
        double best_score = -1;
        std::pair<cv::Point, int> best_result;

        std::cout << "[CircleDetector] 开始处理 " << rois.size() << " 个ROI..." << std::endl;
        
        for (const auto& [roi_img, roi_pos] : rois) {
            // 在ROI中找点
            std::vector<cv::Point> roi_points;
            cv::findNonZero(roi_img, roi_points);
            
            // 调整点的坐标（加上ROI的偏移）
            for (auto& point : roi_points) {
                point.x += roi_pos.x;
                point.y += roi_pos.y;
            }
            
            if (roi_points.size() >= 3) {
                // 拟合圆
                auto result = fitCircle(roi_points);
                if (result.second > 0) {  // 如果找到了有效的圆
                    double score = calculateScore(result.first, result.second, points);
                    if (score > best_score) {
                        best_score = score;
                        best_result = result;
                    }
                }
            }
        }
        
        if (best_score >= min_score_) {
            std::cout << "[CircleDetector] 找到最佳圆: 中心=(" << best_result.first.x 
                      << "," << best_result.first.y << ") 半径=" << best_result.second 
                      << " 得分=" << best_score << std::endl;
            return best_result;
        } else {
            std::cout << "[CircleDetector] 未找到满足条件的圆 (最佳得分=" << best_score 
                      << " < " << min_score_ << ")" << std::endl;
            return {cv::Point(), 0};
        }
        
    } catch (const cv::Exception& e) {
        std::cerr << "[CircleDetector] OpenCV错误: " << e.what() << std::endl;
        return {cv::Point(), 0};
    } catch (const std::exception& e) {
        std::cerr << "[CircleDetector] 标准错误: " << e.what() << std::endl;
        return {cv::Point(), 0};
    } catch (...) {
        std::cerr << "[CircleDetector] 未知错误!" << std::endl;
        return {cv::Point(), 0};
    }
}

cv::Mat CircleDetector::drawResult(bool draw_all_rois) const {
    if (image_.empty() || result_.second == 0) {
        throw std::runtime_error("No image or no circle detected");
    }

    cv::Mat result_image = image_.clone();
    const auto& [center, radius] = result_;
    
    // 绘制圆
    cv::circle(result_image, center, radius, cv::Scalar(0, 255, 0), 2);
    // 绘制圆心
    cv::circle(result_image, center, 5, cv::Scalar(0, 0, 255), -1);

    return result_image;
}

CircleDetector& CircleDetector::saveResult(
    const std::string& output_path, 
    bool draw_all_rois) {
    
    if (result_.second == 0) {
        throw std::runtime_error("No circle detected");
    }
    
    cv::Mat result_image = drawResult(draw_all_rois);
    cv::imwrite(output_path, result_image);
    return *this;
} 