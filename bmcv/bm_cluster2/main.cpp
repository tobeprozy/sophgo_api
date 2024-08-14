#include <string.h>
#include <iostream>
#include "bmcv_api_ext.h"
#include <sys/time.h>

using namespace std;

#define TIME_COST_S(start, end) (((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec)) / 1000000.0)

int main(int argc, char *argv[]) {
    int   row              = 128;
    int   col              = 128;
    float p                = 0.01;
    char  input_path[128]  = {0};
    char  output_path[128] = {0};
    int   min_num_spks     = 2;
    int   max_num_spks     = 8;
    int   num_iter_KNN     = 1;
    int   weight_mode_KNN  = 0;
    int   user_num_spks    = -1;

    FILE          *file;
    struct timeval t1, t2;
    bm_handle_t    handle;
    bm_status_t    ret = bm_dev_request(&handle, 0);

    if (argc != 9 && argc != 10 && argc != 11) {
        printf("usage: %s input output row col p min_num_spks max_num_spks num_iter_KNN [weight_mode_KNN] [num_spks]\n", argv[0]);
        printf("example:\n");
        printf("%s 6k.txt output.txt 128 256 0.01 2 8 2\n", argv[0]);
        printf("[Warning] program exits without run!\n");
        return -1;
    }
    strcpy(input_path, argv[1]);
    strcpy(output_path, argv[2]);
    row          = atoi(argv[3]);
    col          = atoi(argv[4]);
    p            = atof(argv[5]);
    min_num_spks = atoi(argv[6]);
    max_num_spks = atoi(argv[7]);
    num_iter_KNN = atoi(argv[8]);
    if (argc == 10) {
        weight_mode_KNN = atoi(argv[9]);
    }
    if (argc == 11) {
        user_num_spks = atoi(argv[10]);
    }

    float *input_data  = (float *)malloc(row * col * sizeof(float));
    int   *output_data = (int *)malloc(row * max_num_spks * sizeof(int));

    bm_device_mem_t input, output;
    ret = bm_malloc_device_byte(handle, &input, sizeof(float) * row * col);
    if (ret != BM_SUCCESS) {
        bmlib_log("CLUSTER", BMLIB_LOG_ERROR, "bm_malloc_device_byte input error\n");
        goto err0;
    }
    ret = bm_malloc_device_byte(handle, &output, sizeof(float) * row * max_num_spks);
    if (ret != BM_SUCCESS) {
        bmlib_log("CLUSTER", BMLIB_LOG_ERROR, "bm_malloc_device_byte input error\n");
        goto err1;
    }

    file = fopen(input_path, "r");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (fscanf(file, "%f", &input_data[i * col + j]) != 1) {
                if (feof(file)) {
                    printf("End of file reached unexpectedly.\n");
                } else {
                    perror("Error reading the matrix");
                }
                fclose(file);
                free(input_data);
                goto err2;
            }
        }
    }
    printf("read input_data success\n");
    fclose(file);

    ret = bm_memcpy_s2d(handle, input, input_data);
    if (ret != BM_SUCCESS) {
        bmlib_log("CLUSTER", BMLIB_LOG_ERROR, "bm_memcpy_s2d input error\n");
        goto err2;
    }

    printf("------Test Cluster Begin!------\n");
    gettimeofday(&t1, NULL);
    ret = bmcv_cluster(handle,
                       input,
                       output,
                       row,
                       col,
                       p,
                       min_num_spks,
                       max_num_spks,
                       user_num_spks,
                       weight_mode_KNN,
                       num_iter_KNN);

    gettimeofday(&t2, NULL);
    printf("[Final-Total]bmcv_cluster TPU using time: %.2f(s)\n", TIME_COST_S(t1, t2));
    if (ret != BM_SUCCESS) {
        bmlib_log("CLUSTER", BMLIB_LOG_ERROR, "bmcv_cluster error\n");
        goto err2;
    }

    ret = bm_memcpy_d2s(handle, output_data, output);
    if (ret != BM_SUCCESS) {
        bmlib_log("CLUSTER", BMLIB_LOG_ERROR, "bm_memcpy_d2s output error\n");
        goto err2;
    }

    file = fopen(output_path, "w");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }
    for (int i = 0; i < row; i++) {
        int index = i;
        fprintf(file, "%d\n", output_data[index]);
    }
    printf("write output_data success\n");
    fclose(file);
    printf("------Test Cluster Success!------\n");
err2:
    bm_free_device(handle, output);
err1:
    bm_free_device(handle, input);
err0:
    free(input_data);
    free(output_data);
    bm_dev_free(handle);
    return ret;
}