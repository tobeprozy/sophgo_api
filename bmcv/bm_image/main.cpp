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
#include "bmcv_api_ext.h"

using namespace std;

int main(int argc, char* argv[]) {

  int dev_id = 0;
  bm_handle_t bm_handle;
  bm_status_t status = bm_dev_request(&bm_handle, 0);
  bm_image frame1;
  bm_image_create(bm_handle, 460, 1184, FORMAT_BGR_PACKED,                  DATA_TYPE_EXT_1N_BYTE, &frame1);
  bm_image frame2;
  bm_image_create(bm_handle, 460, 1184, FORMAT_BGR_PACKED,                  DATA_TYPE_EXT_1N_BYTE, &frame2);

}
