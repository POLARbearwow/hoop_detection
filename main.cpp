#include "hoop_detector.hpp"
#include <opencv2/opencv.hpp>
#include "MvCameraControl.h"
#include <iostream>
#include <string>
#include <unistd.h>

class HikCamera {
public:
    HikCamera() : handle(nullptr) {
        std::cout << "[HikCamera] 构造函数被调用" << std::endl;
    }
    
    ~HikCamera() {
        std::cout << "[HikCamera] 析构函数被调用" << std::endl;
        closeCamera();
    }

    bool openCamera() {
        int nRet = MV_OK;

        // 初始化SDK
        std::cout << "[HikCamera] 初始化SDK..." << std::endl;
        nRet = MV_CC_Initialize();
        if (MV_OK != nRet) {
            std::cerr << "[HikCamera] Initialize SDK fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }

        // 枚举设备
        std::cout << "[HikCamera] 枚举设备..." << std::endl;
        MV_CC_DEVICE_INFO_LIST stDeviceList;
        memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
        if (MV_OK != nRet) {
            std::cerr << "[HikCamera] Enum Devices fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }
        
        if (stDeviceList.nDeviceNum == 0) {
            std::cerr << "[HikCamera] 未找到任何相机设备!" << std::endl;
            return false;
        }
        std::cout << "[HikCamera] 找到 " << stDeviceList.nDeviceNum << " 个设备" << std::endl;

        // 选择第一个设备
        std::cout << "[HikCamera] 创建相机句柄..." << std::endl;
        nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[0]);
        if (MV_OK != nRet) {
            std::cerr << "[HikCamera] Create Handle fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }

        // 打开设备
        std::cout << "[HikCamera] 打开设备..." << std::endl;
        nRet = MV_CC_OpenDevice(handle);
        if (MV_OK != nRet) {
            std::cerr << "[HikCamera] Open Device fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }

        // 获取相机参数
        MVCC_INTVALUE stParam;
        nRet = MV_CC_GetIntValue(handle, "Width", &stParam);
        if (MV_OK == nRet) {
            std::cout << "[HikCamera] 图像宽度: " << stParam.nCurValue << std::endl;
        }
        nRet = MV_CC_GetIntValue(handle, "Height", &stParam);
        if (MV_OK == nRet) {
            std::cout << "[HikCamera] 图像高度: " << stParam.nCurValue << std::endl;
        }

        // 设置手动曝光
        std::cout << "[HikCamera] 设置相机参数..." << std::endl;
        nRet = MV_CC_SetEnumValue(handle, "ExposureAuto", 0);
        if (MV_OK != nRet) {
            std::cerr << "[HikCamera] Set ExposureAuto fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }

        // 设置曝光时间
        nRet = MV_CC_SetFloatValue(handle, "ExposureTime", 3000.0f);
        if (MV_OK != nRet) {
            std::cerr << "[HikCamera] Set ExposureTime fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }

        // 设置手动增益
        nRet = MV_CC_SetEnumValue(handle, "GainAuto", 0);
        if (MV_OK != nRet) {
            std::cerr << "[HikCamera] Set GainAuto fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }

        // 设置增益值
        nRet = MV_CC_SetFloatValue(handle, "Gain", 23.9f);
        if (MV_OK != nRet) {
            std::cerr << "[HikCamera] Set Gain fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }

        // 设置像素格式为RGB8
        std::cout << "[HikCamera] 设置像素格式..." << std::endl;
        nRet = MV_CC_SetEnumValue(handle, "PixelFormat", PixelType_Gvsp_BGR8_Packed);
        if (MV_OK != nRet) {
            std::cerr << "[HikCamera] Set PixelFormat fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }

        // 开始取流
        std::cout << "[HikCamera] 开始取流..." << std::endl;
        nRet = MV_CC_StartGrabbing(handle);
        if (MV_OK != nRet) {
            std::cerr << "[HikCamera] Start Grabbing fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }

        return true;
    }

    bool getFrame(cv::Mat& frame) {
        if (!handle) {
            std::cerr << "[HikCamera] 相机句柄为空!" << std::endl;
            return false;
        }

        MV_FRAME_OUT stImageInfo = {0};
        memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT));

        std::cout << "[HikCamera] 尝试获取图像..." << std::endl;
        int nRet = MV_CC_GetImageBuffer(handle, &stImageInfo, 1000);
        if (nRet == MV_OK) {
            std::cout << "[HikCamera] 成功获取图像，大小: " 
                      << stImageInfo.stFrameInfo.nWidth << "x" 
                      << stImageInfo.stFrameInfo.nHeight 
                      << " 像素格式: " << stImageInfo.stFrameInfo.enPixelType << std::endl;

            if (stImageInfo.pBufAddr == nullptr) {
                std::cerr << "[HikCamera] 图像缓冲区指针为空!" << std::endl;
                return false;
            }

            try {
                // 转换为OpenCV格式
                cv::Mat rawData(stImageInfo.stFrameInfo.nHeight, 
                              stImageInfo.stFrameInfo.nWidth, 
                              CV_8UC3, 
                              stImageInfo.pBufAddr);
                
                // 复制数据
                rawData.copyTo(frame);

                // 检查复制后的图像
                if (frame.empty()) {
                    std::cerr << "[HikCamera] 复制后的图像为空!" << std::endl;
                    MV_CC_FreeImageBuffer(handle, &stImageInfo);
                    return false;
                }

                std::cout << "[HikCamera] 图像复制成功" << std::endl;
            }
            catch (const cv::Exception& e) {
                std::cerr << "[HikCamera] OpenCV错误: " << e.what() << std::endl;
                MV_CC_FreeImageBuffer(handle, &stImageInfo);
                return false;
            }
            catch (const std::exception& e) {
                std::cerr << "[HikCamera] 标准错误: " << e.what() << std::endl;
                MV_CC_FreeImageBuffer(handle, &stImageInfo);
                return false;
            }
            catch (...) {
                std::cerr << "[HikCamera] 未知错误!" << std::endl;
                MV_CC_FreeImageBuffer(handle, &stImageInfo);
                return false;
            }

            // 释放图像缓存
            nRet = MV_CC_FreeImageBuffer(handle, &stImageInfo);
            if (nRet != MV_OK) {
                std::cerr << "[HikCamera] Free Image Buffer fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
                return false;
            }
            return true;
        }
        else {
            std::cerr << "[HikCamera] 获取图像失败! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }
    }

    void closeCamera() {
        if (handle) {
            std::cout << "[HikCamera] 停止取流..." << std::endl;
            MV_CC_StopGrabbing(handle);
            
            std::cout << "[HikCamera] 关闭设备..." << std::endl;
            MV_CC_CloseDevice(handle);
            
            std::cout << "[HikCamera] 销毁句柄..." << std::endl;
            MV_CC_DestroyHandle(handle);
            handle = nullptr;
        }
        std::cout << "[HikCamera] 终止SDK..." << std::endl;
        MV_CC_Finalize();
    }

private:
    void* handle;
};

int main() {
    try {
        std::cout << "[Main] 程序启动..." << std::endl;
        
        // 创建相机实例
        HikCamera camera;
        if (!camera.openCamera()) {
            std::cerr << "[Main] 相机初始化失败，程序退出" << std::endl;
            return -1;
        }
        std::cout << "[Main] 相机初始化成功！" << std::endl;

        // 创建篮筐检测器实例
        std::cout << "[Main] 创建篮筐检测器..." << std::endl;
        HoopDetector detector(5, 1, 0.7);

        // 创建显示窗口
        std::cout << "[Main] 创建显示窗口..." << std::endl;
        cv::namedWindow("Camera", cv::WINDOW_NORMAL);
        cv::namedWindow("Processing Steps", cv::WINDOW_NORMAL);

        std::cout << "\n=== 开始主循环 ===" << std::endl;
        std::cout << "按'q'键退出程序" << std::endl;

        cv::Mat frame;
        while (true) {
            std::cout << "[Main] 尝试获取新帧..." << std::endl;
            if (camera.getFrame(frame)) {
                if (!frame.empty()) {
                    std::cout << "[Main] 成功获取帧，开始处理..." << std::endl;
                    
                    try {
                        // 处理图像
                        detector.loadImage(frame)
                               .createBinaryImage()
                               .processImage();

                        // 检测篮筐
                        auto [center, radius] = detector.detectCircle();

                        // 显示原始图像
                        cv::imshow("Camera", frame);

                        // 显示处理步骤
                        cv::Mat process_view = detector.showProcess();
                        if (!process_view.empty()) {
                            cv::imshow("Processing Steps", process_view);
                        } else {
                            std::cerr << "[Main] 处理视图为空" << std::endl;
                        }
                    }
                    catch (const cv::Exception& e) {
                        std::cerr << "[Main] OpenCV处理错误: " << e.what() << std::endl;
                    }
                    catch (const std::exception& e) {
                        std::cerr << "[Main] 处理错误: " << e.what() << std::endl;
                    }
                    catch (...) {
                        std::cerr << "[Main] 未知处理错误!" << std::endl;
                    }

                    // 检查键盘输入
                    char key = cv::waitKey(1);
                    if (key == 'q') {
                        std::cout << "[Main] 检测到退出指令，准备退出..." << std::endl;
                        break;
                    }
                } else {
                    std::cerr << "[Main] 警告: 获取到空图像" << std::endl;
                }
            } else {
                std::cerr << "[Main] 获取图像失败" << std::endl;
                usleep(100000);  // 失败时等待100ms
            }
        }

        std::cout << "[Main] 程序结束，开始清理资源..." << std::endl;
        cv::destroyAllWindows();
        std::cout << "[Main] 程序正常退出" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[Main] 发生错误: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[Main] 发生未知错误!" << std::endl;
        return 1;
    }

    return 0;
} 