#!/bin/sh
# clean up CMake artifacts
dir="${1:-.}"
find "$dir" \( \
    -name CMakeFiles -type d -exec rm -rf {} + \
\) -o \( \
    -name cmake_install.cmake -exec rm {} + \
\) -o \( \
    -name CMakeCache.txt -exec rm {} + \
\) -o \( \
    -name Makefile -exec rm {} + \
\) -o \( \
    -name '*.dir' -type d -exec rm -rf {} + \
\)
