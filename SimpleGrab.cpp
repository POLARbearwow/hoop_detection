#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "MvCameraControl.h"

class HikCamera {
public:
    HikCamera() : handle(nullptr) {}
    ~HikCamera() { closeCamera(); }

    bool openCamera() {
        int nRet = MV_OK;

        // 初始化SDK
        nRet = MV_CC_Initialize();
        if (MV_OK != nRet) {
            printf("Initialize SDK fail! nRet [0x%x]\n", nRet);
            return false;
        }

        // 枚举设备
        MV_CC_DEVICE_INFO_LIST stDeviceList;
        memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
        if (MV_OK != nRet || stDeviceList.nDeviceNum == 0) {
            printf("Enum Devices fail! nRet [0x%x]\n", nRet);
            return false;
        }

        // 选择第一个设备
        nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[0]);
        if (MV_OK != nRet) {
            printf("Create Handle fail! nRet [0x%x]\n", nRet);
            return false;
        }

        // 打开设备
        nRet = MV_CC_OpenDevice(handle);
        if (MV_OK != nRet) {
            printf("Open Device fail! nRet [0x%x]\n", nRet);
            return false;
        }

        // 设置手动曝光
        nRet = MV_CC_SetEnumValue(handle, "ExposureAuto", 0);
        if (MV_OK != nRet) {
            printf("Set ExposureAuto fail! nRet [0x%x]\n", nRet);
            return false;
        }

        // 设置曝光时间
        nRet = MV_CC_SetFloatValue(handle, "ExposureTime", 3000.0f);
        if (MV_OK != nRet) {
            printf("Set ExposureTime fail! nRet [0x%x]\n", nRet);
            return false;
        }

        // 设置手动增益
        nRet = MV_CC_SetEnumValue(handle, "GainAuto", 0);
        if (MV_OK != nRet) {
            printf("Set GainAuto fail! nRet [0x%x]\n", nRet);
            return false;
        }

        // 设置增益值
        nRet = MV_CC_SetFloatValue(handle, "Gain", 23.9f);
        if (MV_OK != nRet) {
            printf("Set Gain fail! nRet [0x%x]\n", nRet);
            return false;
        }

        // 设置像素格式为RGB8
        nRet = MV_CC_SetEnumValue(handle, "PixelFormat", PixelType_Gvsp_BGR8_Packed);
        if (MV_OK != nRet) {
            printf("Set PixelFormat fail! nRet [0x%x]\n", nRet);
            return false;
        }

        // 开始取流
        nRet = MV_CC_StartGrabbing(handle);
        if (MV_OK != nRet) {
            printf("Start Grabbing fail! nRet [0x%x]\n", nRet);
            return false;
        }

        return true;
    }

    bool getFrame(cv::Mat& frame) {
        MV_FRAME_OUT stImageInfo = {0};
        memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT));

        int nRet = MV_CC_GetImageBuffer(handle, &stImageInfo, 1000);
        if (nRet == MV_OK) {
            // 转换为OpenCV格式
            cv::Mat rawData(stImageInfo.stFrameInfo.nHeight, 
                          stImageInfo.stFrameInfo.nWidth, 
                          CV_8UC3, 
                          stImageInfo.pBufAddr);
            
            // 复制数据
            rawData.copyTo(frame);

            // 释放图像缓存
            nRet = MV_CC_FreeImageBuffer(handle, &stImageInfo);
            if (nRet != MV_OK) {
                printf("Free Image Buffer fail! nRet [0x%x]\n", nRet);
                return false;
            }
            return true;
        }
        return false;
    }

    void closeCamera() {
        if (handle) {
            MV_CC_StopGrabbing(handle);
            MV_CC_CloseDevice(handle);
            MV_CC_DestroyHandle(handle);
            handle = nullptr;
        }
        MV_CC_Finalize();
    }

private:
    void* handle;
};

int main() {
    HikCamera camera;
    
    if (!camera.openCamera()) {
        printf("Failed to open camera!\n");
        return -1;
    }

    printf("Camera opened successfully!\n");
    printf("Press 'q' to exit\n");

    cv::Mat frame;
    while (true) {
        if (camera.getFrame(frame)) {
            cv::imshow("Hikvision Camera", frame);
            
            // 按q退出
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
    }

    cv::destroyAllWindows();
    return 0;
} 