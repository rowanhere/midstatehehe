#!/usr/bin/env bash
set -euo pipefail

make clean
make

name="midstate-cuda-miner-linux-x86_64-cuda12"
rm -rf "dist/$name" "$name.tar.gz" SHA256SUMS.txt
mkdir -p "dist/$name"
cp midstate-cuda-miner README.md "dist/$name/"
tar -C dist -czf "$name.tar.gz" "$name"
sha256sum "$name.tar.gz" > SHA256SUMS.txt

echo "Built $name.tar.gz"
cat SHA256SUMS.txt
