//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <string.h>
#include <iostream>
#include "ff_decode.hpp"
#include "bmcv_api_ext_c.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;



void create_and_convert_bytes() {
    size_t byte_size = 10; // 假设我们想要创建包含10个字节的数组
    unsigned char* output_ptr = static_cast<unsigned char*>(malloc(byte_size));

    // 为了测试，我们可以手动填充这个数组
    for (size_t i = 0; i < byte_size; ++i) {
        output_ptr[i] = static_cast<unsigned char>(i);
    }

    // 将数据转换为 Python 字节对象
    py::bytes result(reinterpret_cast<const char*>(output_ptr), byte_size);

    // 打印 Python 字节对象，需要先转换为 std::string
    std::string result_str = static_cast<std::string>(result);
    std::cout << "Python bytes object: ";
    for (char c : result_str) {
        std::cout << std::hex << static_cast<int>(c) << " ";
    }
    std::cout << std::endl;

    // 清理分配的内存
    free(output_ptr);
}


bool bytes_to_bm_image(
    bm_handle_t bm_handle,
    const std::vector<unsigned char>& bytes,
    int width, int height,
    bm_image& output_image)
{
    bm_status_t ret;
    // 根据提供的宽度和高度计算 YUV420P 格式的字节大小
    int y_size = width * height;
    int uv_size = (width / 2) * (height / 2);
    int byte_size = y_size + uv_size * 2;

    // 检查提供的字节数据是否符合预期大小
    if (bytes.size() != byte_size) {
        // 处理错误：字节大小与预期不符
        return false;
    }

    // 创建 bm_image 对象
    ret = bm_image_create(bm_handle, height, width, FORMAT_YUV420P, DATA_TYPE_EXT_1N_BYTE, &output_image);
    if (ret != BM_SUCCESS) {
        // 处理错误：bm_image 创建失败
        return false;
    }

    // 分配设备内存
    #if (BMCV_VERSION_MAJOR == 2) && !defined(BMCV_VERSION_MINOR)
        ret = bm_image_alloc_dev_mem_heap_mask(output_image, 2);
    #else
        ret = bm_image_alloc_dev_mem_heap_mask(output_image, 6);
    #endif
        if (ret != BM_SUCCESS) {
            // 处理错误：设备内存分配失败
            bm_image_destroy(output_image);
            return false;
        }

    // 准备指向 Y, U, V 数据的指针
    void* src_plane[4];
    src_plane[0] = (void*)bytes.data();
    src_plane[1] = (void*)(bytes.data() + y_size);
    src_plane[2] = (void*)(bytes.data() + y_size + uv_size);
    src_plane[3] = nullptr; // YUV420P 不使用第四个平面

    // 将字节数据复制到设备内存
    ret = bm_image_copy_host_to_device(output_image, src_plane);
    if (ret != BM_SUCCESS) {
        // 处理错误：数据复制失败
        bm_image_destroy(output_image);
        return false;
    }

    return true;
}

bool bm_image_to_bytes(bm_handle_t bm_handle, const bm_image& frame, std::vector<unsigned char>& bytes) {
    int width = frame.width;
    int height = frame.height;

    int y_size = width * height;
    int uv_size = (width / 2) * (height / 2);

    int byte_size = y_size + 2 * uv_size; // YUV420P 格式的总字节数

    unsigned char* output_ptr = (unsigned char *)malloc(byte_size);
    if (!output_ptr) {
        // 处理内存分配失败
        return false;
    }

    void* out_ptr[4] = {
        (void*)output_ptr,
        (void*)(output_ptr + y_size),
        (void*)(output_ptr + y_size + uv_size),
        nullptr // YUV420P 不使用第四个平面
    };

    // 将 bm_image 保存为 BMP 文件（可选）
    // bm_image_write_to_bmp(frame, "frame.bmp");

    // 从设备复制图像数据到主机
    bm_status_t status = bm_image_copy_device_to_host(frame, (void **)out_ptr);
    if (status != BM_SUCCESS) {
        // 处理复制失败
        free(output_ptr);
        return false;
    }

    // 使用 C++ 表示字节对象
    bytes.assign(output_ptr, output_ptr + byte_size);

    // 释放临时分配的内存
    free(output_ptr);

    return true;
}



int main(int argc, char* argv[]) {
  cout << "nihao!!" << endl;

  string img_file = "../1920x1080_yuvj420.jpg";
  int dev_id = 0;
  bm_handle_t bm_handle;
  bm_status_t status = bm_dev_request(&bm_handle, 0);

  bm_image bmimg;
  picDec(bm_handle, img_file.c_str(), bmimg);

  bm_image frame;
  bm_image_create(bm_handle, bmimg.height, bmimg.width, FORMAT_YUV420P,
                  bmimg.data_type, &frame);
  bmcv_image_storage_convert(bm_handle, 1, &bmimg, &frame);

  int width = frame.width;
  int height = frame.height;
<<<<<<< HEAD
  std::vector<unsigned char> byte_data;
  bm_image_to_bytes(bm_handle, frame, byte_data);

  bm_image frame1;
  bytes_to_bm_image(bm_handle, byte_data, width, height, frame1);

  bm_image_write_to_bmp(frame, "frame.bmp");
  bm_image_write_to_bmp(frame1, "frame1.bmp");

  const char data[] = "Hello, World!";
  size_t data_length = sizeof(data); // 包括结尾的 null 字符
  // 创建一个 Python 字节对象
  // py::bytes result(data, data_length);

  // 将数据转换为 Python 字节对象
  // py::bytes result2(reinterpret_cast<const char*>(output_ptr), byte_size);

  // create_and_convert_bytes();
  std::cout << "result: "<< std::endl;
  std::cout << "result2: "<< std::endl;

}
