model_transform \
    --model_name bytetrack_s \
    --model_def ../bytetrack_s.onnx \
    --input_shapes [[1,3,608,1088]] \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --test_input ./dog.jpg \
    --test_result bytetrack_s_top_outputs.npz \
    --mlir bytetrack_s.mlir

model_deploy \
    --mlir bytetrack_s.mlir \
    --quantize F32 \
    --processor bm1684 \
    --test_input bytetrack_s_in_f32.npz \
    --test_reference bytetrack_s_top_outputs.npz \
    --model bytetrack_s_v2_fp32.bmodel