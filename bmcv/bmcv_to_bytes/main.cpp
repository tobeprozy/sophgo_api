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

  int y_size = width * height;
  int uv_size = (width / 2) * (height / 2);

  int byte_size = y_size + 2 * uv_size; // YUV420P 格式的总字节数

  unsigned char* output_ptr = (unsigned char *)malloc(byte_size);
  void* out_ptr[4] = {(void*)output_ptr,
                      (void*)(output_ptr + y_size),
                      (void*)(output_ptr + y_size + uv_size),
                      (void*)(output_ptr + y_size + uv_size * 2)};


  bm_image_copy_device_to_host(frame, (void **)out_ptr);

  const char data[] = "Hello, World!";
  size_t data_length = sizeof(data); // 包括结尾的 null 字符
  // 创建一个 Python 字节对象
  py::bytes result(data, data_length);

  // 将数据转换为 Python 字节对象
  // py::bytes result2(reinterpret_cast<const char*>(output_ptr), byte_size);

  //使用c++表示字节对象
  std::vector<unsigned char> bytes(output_ptr, output_ptr + byte_size);

  // create_and_convert_bytes();



  std::cout << "result: "<< std::endl;
  std::cout << "result2: "<< std::endl;

}
