#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "hik_camera.hpp"
#include "hoop_detector.hpp"
#include "camera_calibrator.hpp"
#include "image_saver.hpp"
#include "MvCameraControl.h"
#include <unistd.h>

// class HikCamera {
// public:
//     HikCamera() : handle(nullptr) {
//         std::cout << "[HikCamera] 构造函数被调用" << std::endl;
//     }
    
//     ~HikCamera() {
//         std::cout << "[HikCamera] 析构函数被调用" << std::endl;
//         closeCamera();
//     }

//     bool openCamera() {
//         int nRet = MV_OK;

//         // 初始化SDK
//         std::cout << "[HikCamera] 初始化SDK..." << std::endl;
//         nRet = MV_CC_Initialize();
//         if (MV_OK != nRet) {
//             std::cerr << "[HikCamera] Initialize SDK fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
//             return false;
//         }

//         // 枚举设备
//         std::cout << "[HikCamera] 枚举设备..." << std::endl;
//         MV_CC_DEVICE_INFO_LIST stDeviceList;
//         memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
//         nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
//         if (MV_OK != nRet) {
//             std::cerr << "[HikCamera] Enum Devices fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
//             return false;
//         }
        
//         if (stDeviceList.nDeviceNum == 0) {
//             std::cerr << "[HikCamera] 未找到任何相机设备!" << std::endl;
//             return false;
//         }
//         std::cout << "[HikCamera] 找到 " << stDeviceList.nDeviceNum << " 个设备" << std::endl;

//         // 选择第一个设备
//         std::cout << "[HikCamera] 创建相机句柄..." << std::endl;
//         nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[0]);
//         if (MV_OK != nRet) {
//             std::cerr << "[HikCamera] Create Handle fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
//             return false;
//         }

//         // 打开设备
//         std::cout << "[HikCamera] 打开设备..." << std::endl;
//         nRet = MV_CC_OpenDevice(handle);
//         if (MV_OK != nRet) {
//             std::cerr << "[HikCamera] Open Device fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
//             return false;
//         }

//         // 获取相机参数
//         MVCC_INTVALUE stParam;
//         nRet = MV_CC_GetIntValue(handle, "Width", &stParam);
//         if (MV_OK == nRet) {
//             std::cout << "[HikCamera] 图像宽度: " << stParam.nCurValue << std::endl;
//         }
//         nRet = MV_CC_GetIntValue(handle, "Height", &stParam);
//         if (MV_OK == nRet) {
//             std::cout << "[HikCamera] 图像高度: " << stParam.nCurValue << std::endl;
//         }

//         // 设置手动曝光
//         std::cout << "[HikCamera] 设置相机参数..." << std::endl;
//         nRet = MV_CC_SetEnumValue(handle, "ExposureAuto", 0);
//         if (MV_OK != nRet) {
//             std::cerr << "[HikCamera] Set ExposureAuto fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
//             return false;
//         }

//         // 设置曝光时间
//         nRet = MV_CC_SetFloatValue(handle, "ExposureTime", 3000.0f);
//         if (MV_OK != nRet) {
//             std::cerr << "[HikCamera] Set ExposureTime fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
//             return false;
//         }

//         // 设置手动增益
//         nRet = MV_CC_SetEnumValue(handle, "GainAuto", 0);
//         if (MV_OK != nRet) {
//             std::cerr << "[HikCamera] Set GainAuto fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
//             return false;
//         }

//         // 设置增益值
//         nRet = MV_CC_SetFloatValue(handle, "Gain", 23.9f);
//         if (MV_OK != nRet) {
//             std::cerr << "[HikCamera] Set Gain fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
//             return false;
//         }

//         // 设置像素格式为RGB8
//         std::cout << "[HikCamera] 设置像素格式..." << std::endl;
//         nRet = MV_CC_SetEnumValue(handle, "PixelFormat", PixelType_Gvsp_BGR8_Packed);
//         if (MV_OK != nRet) {
//             std::cerr << "[HikCamera] Set PixelFormat fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
//             return false;
//         }

//         // 开始取流
//         std::cout << "[HikCamera] 开始取流..." << std::endl;
//         nRet = MV_CC_StartGrabbing(handle);
//         if (MV_OK != nRet) {
//             std::cerr << "[HikCamera] Start Grabbing fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
//             return false;
//         }

//         return true;
//     }

//     bool getFrame(cv::Mat& frame) {
//         if (!handle) {
//             std::cerr << "[HikCamera] 相机句柄为空!" << std::endl;
//             return false;
//         }

//         MV_FRAME_OUT stImageInfo = {0};
//         memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT));

//         std::cout << "[HikCamera] 尝试获取图像..." << std::endl;
//         int nRet = MV_CC_GetImageBuffer(handle, &stImageInfo, 1000);
//         if (nRet == MV_OK) {
//             std::cout << "[HikCamera] 成功获取图像，大小: " 
//                       << stImageInfo.stFrameInfo.nWidth << "x" 
//                       << stImageInfo.stFrameInfo.nHeight 
//                       << " 像素格式: " << stImageInfo.stFrameInfo.enPixelType << std::endl;

//             if (stImageInfo.pBufAddr == nullptr) {
//                 std::cerr << "[HikCamera] 图像缓冲区指针为空!" << std::endl;
//                 return false;
//             }

//             cv::Mat rawData(stImageInfo.stFrameInfo.nHeight, 
//                               stImageInfo.stFrameInfo.nWidth, 
//                               CV_8UC3, 
//                               stImageInfo.pBufAddr);
                
//                 // 复制数据
//                 rawData.copyTo(frame);

//                 // 检查复制后的图像
//                 if (frame.empty()) {
//                     std::cerr << "[HikCamera] 复制后的图像为空!" << std::endl;
//                     MV_CC_FreeImageBuffer(handle, &stImageInfo);
//                     return false;
//                 }

//                 std::cout << "[HikCamera] 图像复制成功" << std::endl;
//             }
//             catch (const cv::Exception& e) {
//                 std::cerr << "[HikCamera] OpenCV错误: " << e.what() << std::endl;
//                 MV_CC_FreeImageBuffer(handle, &stImageInfo);
//                 return false;
//             }
//             catch (const std::exception& e) {
//                 std::cerr << "[HikCamera] 标准错误: " << e.what() << std::endl;
//                 MV_CC_FreeImageBuffer(handle, &stImageInfo);
//                 return false;
//             }
//             catch (...) {
//                 std::cerr << "[HikCamera] 未知错误!" << std::endl;
//                 MV_CC_FreeImageBuffer(handle, &stImageInfo);
//                 return false;
//             }

//             // 释放图像缓存
//             nRet = MV_CC_FreeImageBuffer(handle, &stImageInfo);
//             if (nRet != MV_OK) {
//                 std::cerr << "[HikCamera] Free Image Buffer fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
//                 return false;
//             }
//             return true;
//         }
//         else {
//             std::cerr << "[HikCamera] 获取图像失败! nRet [0x" << std::hex << nRet << "]" << std::endl;
//             return false;
//         }
//     }

//     void closeCamera() {
//         if (handle) {
//             std::cout << "[HikCamera] 停止取流..." << std::endl;
//             MV_CC_StopGrabbing(handle);
            
//             std::cout << "[HikCamera] 关闭设备..." << std::endl;
//             MV_CC_CloseDevice(handle);
            
//             std::cout << "[HikCamera] 销毁句柄..." << std::endl;
//             MV_CC_DestroyHandle(handle);
//             handle = nullptr;
//         }
//         std::cout << "[HikCamera] 终止SDK..." << std::endl;
//         MV_CC_Finalize();
//     }

// private:
//     void* handle;
// };

void printUsage() {
    std::cout << "使用方法:\n"
              << "1. 摄像头模式: ./hoop_detecor --camera\n"
              << "2. 图片模式: ./hoop_detecor --image <图片路径>\n"
              << "3. 标定模式: ./hoop_detecor --calibrate <棋盘格宽> <棋盘格高> <方格大小(mm)> <图片目录>\n"
              << "4. 验证标定: ./hoop_detecor --verify <标定文件> <测试图片>\n"
              << "5. 保存图片: ./hoop_detecor --save [保存目录]\n"
              << "\n示例:\n"
              << "1. 摄像头模式: ./hoop_detecor --camera\n"
              << "2. 图片模式: ./hoop_detecor --image test.jpg\n"
              << "3. 标定模式: ./hoop_detecor --calibrate 9 6 20 ./chess_images/\n"
              << "4. 验证标定: ./hoop_detecor --verify camera_params.yml test.jpg\n"
              << "5. 保存图片: ./hoop_detecor --save ./captured_images" << std::endl;
}

// 处理单张图片的函数
bool processImage(const std::string& imagePath, HoopDetector& detector) {
    std::cout << "[Main] 正在处理图片: " << imagePath << std::endl;
    
    cv::Mat frame = cv::imread(imagePath);
    if (frame.empty()) {
        std::cerr << "[Main] 无法读取图片: " << imagePath << std::endl;
        return false;
    }

    try {
        // 处理图像
        detector.loadImage(frame)
               .createBinaryImage()
               .processImage();

        // 检测篮筐
        auto [center, radius] = detector.detectCircle();

        // 显示原始图像
        cv::imshow("Image", frame);

        // 显示处理步骤
        cv::Mat process_view = detector.showProcess();
        if (!process_view.empty()) {
            cv::imshow("Processing Steps", process_view);
        }

        std::cout << "[Main] 按任意键继续..." << std::endl;
        cv::waitKey(0);
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[Main] 处理错误: " << e.what() << std::endl;
        return false;
    }
}

// 处理摄像头的函数
void processCameraStream(HikCamera& camera, HoopDetector& detector) {
    cv::Mat frame;
    char key = 0;
    bool running = true;
    int frame_count = 0;
    const int LOG_INTERVAL = 30; // 每30帧输出一次结果

    while (running && key != 'q') {
        if (!camera.getFrame(frame)) {
            // std::cerr << "[Main] 获取图像失败" << std::endl;
            continue;
        }

        // 分开调用loadImage和processImage
        detector.loadImage(frame);
        detector.createBinaryImage();
        detector.processImage();

        // 检测篮筐
        auto [center, radius] = detector.detectCircle();

        // 显示结果
        cv::imshow("Camera", frame);
        cv::imshow("Processing Steps", detector.showProcess());
        
        // 获取PnP结果并定期输出
        if (frame_count % LOG_INTERVAL == 0 && radius > 0) {
            // 计算位姿
            cv::Vec3f pose = detector.solvePnP(center, radius);
            
            // 输出位姿信息
            std::cout << "\n=== 篮筐位姿 (第 " << frame_count << " 帧) ===" << std::endl;
            std::cout << "位置 (米):" << std::endl;
            std::cout << "X: " << std::fixed << std::setprecision(3) << pose[0] << std::endl;
            std::cout << "Y: " << std::fixed << std::setprecision(3) << pose[1] << std::endl;
            std::cout << "Z: " << std::fixed << std::setprecision(3) << pose[2] << std::endl;
        }

        frame_count++;
        key = cv::waitKey(1);
    }
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "[Main] 程序启动..." << std::endl;

        // 检查命令行参数
        if (argc < 2) {
            printUsage();
            return -1;
        }

        std::string mode(argv[1]);
        
        if (mode == "--camera") {
            // 摄像头模式
            HikCamera camera;
            if (!camera.openCamera()) {
                std::cerr << "[Main] 相机初始化失败" << std::endl;
                return -1;
            }
            std::cout << "[Main] 相机初始化成功！" << std::endl;

            // 创建篮筐检测器实例
            std::cout << "[Main] 创建篮筐检测器..." << std::endl;
            HoopDetector detector(5, 1, 0.7);

            // 尝试加载相机参数
            cv::FileStorage fs("camera_params.yml", cv::FileStorage::READ);
            if (fs.isOpened()) {
                cv::Mat camera_matrix, dist_coeffs;
                fs["camera_matrix"] >> camera_matrix;
                fs["distortion_coefficients"] >> dist_coeffs;
                detector.setCameraParams(camera_matrix, dist_coeffs);
                std::cout << "[Main] 已加载相机标定参数" << std::endl;
            } else {
                std::cout << "[Main] 未找到相机标定文件，将使用默认参数" << std::endl;
            }

            // 创建显示窗口
            cv::namedWindow("Camera", cv::WINDOW_NORMAL);
            cv::namedWindow("Processing Steps", cv::WINDOW_NORMAL);

            std::cout << "\n=== 开始摄像头模式 ===" << std::endl;
            std::cout << "按'q'键退出程序" << std::endl;

            processCameraStream(camera, detector);
        }
        else if (mode == "--image") {
            if (argc < 3) {
                std::cerr << "[Main] 错误：未指定图片路径" << std::endl;
                printUsage();
                return -1;
            }

            std::string imagePath(argv[2]);
            
            // 创建篮筐检测器实例
            std::cout << "[Main] 创建篮筐检测器..." << std::endl;
            HoopDetector detector(5, 1, 0.7);

            // 尝试加载相机参数
            cv::FileStorage fs("camera_params.yml", cv::FileStorage::READ);
            if (fs.isOpened()) {
                cv::Mat camera_matrix, dist_coeffs;
                fs["camera_matrix"] >> camera_matrix;
                fs["distortion_coefficients"] >> dist_coeffs;
                detector.setCameraParams(camera_matrix, dist_coeffs);
                std::cout << "[Main] 已加载相机标定参数" << std::endl;
            } else {
                std::cout << "[Main] 未找到相机标定文件，将使用默认参数" << std::endl;
            }

            // 创建显示窗口
            cv::namedWindow("Image", cv::WINDOW_NORMAL);
            cv::namedWindow("Processing Steps", cv::WINDOW_NORMAL);

            std::cout << "\n=== 开始图片处理模式 ===" << std::endl;
            
            if (!processImage(imagePath, detector)) {
                std::cerr << "[Main] 图片处理失败" << std::endl;
                return -1;
            }
        }
        else if (mode == "--calibrate") {
            if (argc != 6) {
                std::cerr << "[Main] 错误: 标定模式参数不正确" << std::endl;
                printUsage();
                return -1;
            }

            int board_width = std::stoi(argv[2]);
            int board_height = std::stoi(argv[3]);
            float square_size = std::stof(argv[4]);
            std::string image_dir(argv[5]);

            CameraCalibrator calibrator;
            if (!calibrator.calibrateFromImages(image_dir, board_width, board_height, square_size)) {
                std::cerr << "[Main] 标定失败" << std::endl;
                return -1;
            }

            if (!calibrator.saveCalibrationResult("camera_params.yml")) {
                std::cerr << "[Main] 保存标定结果失败" << std::endl;
                return -1;
            }
        }
        else if (mode == "--verify") {
            if (argc != 4) {
                std::cerr << "[Main] 错误: 验证模式参数不正确" << std::endl;
                printUsage();
                return -1;
            }

            std::string calib_file(argv[2]);
            std::string test_image(argv[3]);

            CameraCalibrator calibrator;
            if (!calibrator.verifyCalibration(calib_file, test_image)) {
                std::cerr << "[Main] 验证失败" << std::endl;
                return -1;
            }
        }
        else if (mode == "--save") {
            std::string save_dir = (argc > 2) ? argv[2] : "saved_images";
            
            ImageSaver saver(save_dir);
            if (!saver.run()) {
                std::cerr << "[Main] 图片保存模式运行失败" << std::endl;
                return -1;
            }
        }
        else {
            std::cerr << "[Main] 未知的运行模式: " << mode << std::endl;
            printUsage();
            return -1;
        }

        std::cout << "[Main] 程序结束，开始清理资源..." << std::endl;
        cv::destroyAllWindows();
        std::cout << "[Main] 程序正常退出" << std::endl;

    } catch (const cv::Exception& e) {
        std::cerr << "[Main] OpenCV错误: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[Main] 发生错误: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[Main] 发生未知错误!" << std::endl;
        return 1;
    }

    return 0;
} 