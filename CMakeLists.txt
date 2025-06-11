cmake_minimum_required(VERSION 3.10)
project(hoop_detector)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV包
find_package(OpenCV REQUIRED)

# 添加海康相机SDK的头文件路径
include_directories(/opt/MVS/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 添加所有源文件
add_executable(hoop_detector
    main.cpp
    hoop_detector.cpp
    circle_detector.cpp
)

# 链接OpenCV库和海康相机SDK
target_link_libraries(hoop_detector 
    ${OpenCV_LIBS}
    /opt/MVS/lib/64/libMvCameraControl.so
)

# 包含头文件目录
target_include_directories(hoop_detector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
) 