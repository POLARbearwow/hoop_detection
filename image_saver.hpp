#pragma once
#include <opencv2/opencv.hpp>
#include "hik_camera.hpp"
#include <string>

class ImageSaver {
public:
    ImageSaver(const std::string& save_dir = "saved_images");
    ~ImageSaver() = default;

    // 运行图片保存程序
    bool run();

    // 设置保存目录
    void setSaveDirectory(const std::string& dir) { save_dir_ = dir; }
    
    // 获取保存目录
    std::string getSaveDirectory() const { return save_dir_; }

private:
    // 创建保存目录
    bool createSaveDirectory();
    
    // 生成文件名
    std::string generateFilename() const;

private:
    std::string save_dir_;      // 保存目录
    size_t image_counter_;      // 图片计数器
    HikCamera camera_;          // 相机对象
}; 