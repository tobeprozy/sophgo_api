#!/bin/bash
res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

# datasets
if [ ! -d "datasets" ]; 
then
    mkdir datasets
    pushd datasets
    wget https://gofile.me/7cdg0/EmGDQdecU/input_data.npz
    wget https://gofile.me/7cdg0/EmGDQdecU/matmul.onnx
    wget https://gofile.me/7cdg0/EmGDQdecU/matmul.bmodel

    mv  input_data.npz ../datasets
    mv matmul.onnx ../datasets
    mv matmul.bmodel ../datasets
    popd
    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# if [ ! -d "../models" ]; 
# then
#     mkdir ../models
#     pushd ../models
#     popd
#     echo "models download!"
# else
#     echo "Models folder exist! Remove it if you need to update."
# fi



