#!/bin/bash
cmake -DCMAKE_TOOLCHAIN_FILE=~/hack/vcpkg/scripts/buildsystems/vcpkg.cmake .
make clean
make llama -j8
make quantize -j8
