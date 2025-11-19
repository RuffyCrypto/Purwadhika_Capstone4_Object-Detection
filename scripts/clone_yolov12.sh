#!/usr/bin/env bash
set -e
REPO_DIR=yolov12
if [ -d "$REPO_DIR" ]; then
  echo "yolov12 already exists."
  exit 0
fi
git clone https://github.com/sunsmarterjie/yolov12.git $REPO_DIR
echo "Cloned yolov12 into $REPO_DIR"
