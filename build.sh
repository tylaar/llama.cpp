#!/bin/bash
cmake -DCMAKE_TOOLCHAIN_FILE=/Users/yifengyu/hack/vcpkg/scripts/buildsystems/vcpkg.cmake .
make clean
make llama
