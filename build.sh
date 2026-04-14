#!/bin/bash
set -euo pipefail

ARCH=$(uname -m)   # x86_64 or aarch64
BUILD_DIR="build/${ARCH}"

echo "Building people_counter for ${ARCH}..."
mkdir -p "${BUILD_DIR}"
cmake -H. -B"${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -- -j"$(nproc)"

if [[ -f "hailort.log" ]]; then
    rm -f hailort.log
fi

echo ""
echo "Build complete: ${BUILD_DIR}/people_counter"
echo ""
echo "Usage:"
echo "  ${BUILD_DIR}/people_counter --net <model.hef> --input rpi"
echo "  ${BUILD_DIR}/people_counter --net <model.hef> --input usb"
echo "  ${BUILD_DIR}/people_counter --net <model.hef> --input <video.mp4>"
echo "  ${BUILD_DIR}/people_counter --net <model.hef> --input rpi --config /path/to/config.yaml"
