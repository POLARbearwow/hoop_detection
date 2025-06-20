#include "image_saver.hpp"
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <ctime>

ImageSaver::ImageSaver(const std::string& save_dir)
    : save_dir_(save_dir)
    , image_counter_(0)
{
    std::cout << "[ImageSaver] 初始化，保存目录: " << save_dir_ << std::endl;
}

bool ImageSaver::createSaveDirectory() {
    try {
        if (!std::filesystem::exists(save_dir_)) {
            if (!std::filesystem::create_directories(save_dir_)) {
                std::cerr << "[ImageSaver] 创建目录失败: " << save_dir_ << std::endl;
                return false;
            }
            std::cout << "[ImageSaver] 创建目录: " << save_dir_ << std::endl;
        }
        return true;
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "[ImageSaver] 文件系统错误: " << e.what() << std::endl;
        return false;
    }
}

std::string ImageSaver::generateFilename() const {
    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << save_dir_ << "/";
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
    ss << "_" << std::setfill('0') << std::setw(4) << image_counter_;
    ss << ".jpg";

    return ss.str();
}

bool ImageSaver::run() {
    if (!createSaveDirectory()) {
        return false;
    }

    // 初始化相机
    if (!camera_.openCamera()) {
        std::cerr << "[ImageSaver] 相机初始化失败" << std::endl;
        return false;
    }

    cv::namedWindow("Camera", cv::WINDOW_NORMAL);
    std::cout << "\n=== 图片保存模式 ===" << std::endl;
    std::cout << "按's'保存图片" << std::endl;
    std::cout << "按'q'退出程序" << std::endl;

    cv::Mat frame;
    char key;
    bool running = true;

    while (running) {
        // 获取图像
        if (!camera_.getFrame(frame)) {
            std::cerr << "[ImageSaver] 获取图像失败" << std::endl;
            continue;
        }

        // 显示图像
        cv::imshow("Camera", frame);
        key = cv::waitKey(1);

        // 处理按键
        switch (key) {
            case 's':
            case 'S': {
                std::string filename = generateFilename();
                if (cv::imwrite(filename, frame)) {
                    std::cout << "[ImageSaver] 保存图片: " << filename << std::endl;
                    image_counter_++;
                } else {
                    std::cerr << "[ImageSaver] 保存图片失败: " << filename << std::endl;
                }
                break;
            }
            case 'q':
            case 'Q':
            case 27: // ESC
                running = false;
                break;
        }
    }

    cv::destroyWindow("Camera");
    return true;
} 