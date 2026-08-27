#!/bin/sh

# modify the GGML_FLAGS below as needed, they are passed through to GGML.
# refer to the full list of options at: ggml/CMakeLists.txt.
# these default flags are for macOS Apple Silicon devices.
GGML_FLAGS="-DGGML_NATIVE=1 -DARM_NATIVE_FLAG=-mcpu=native -DGGML_OPENMP=0 -DGGML_METAL=1"

rm -rf ./build
cmake -B ./build -DCMAKE_BUILD_TYPE=Release ${GGML_FLAGS}
cmake --build ./build --config Release -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"