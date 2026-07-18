# midstate-cuda-miner

Standalone NVIDIA/CUDA miner for the open Midstate Stratum pool protocol.

This is separate from the official locked TRACE HTTP miner. It targets a pool
started with the open Midstate node:

```bash
midstate pool --bind-addr 0.0.0.0:3333 ...
```

## Release Binary

Release binaries are built as CUDA fat binaries for:

```text
GTX 1070 Ti: sm_61
RTX 3090:    sm_86
RTX 4090:    sm_89
RTX 5090:    sm_120
```

On a mining rig, the release binary should only need the NVIDIA driver. You do
not need to install the CUDA toolkit or Vulkan.

## Build From Source

Builds require CUDA 12.8+ because RTX 5090 support needs `sm_120`:

```bash
sudo apt update
sudo apt install -y build-essential nvidia-cuda-toolkit
make
```

If Clore already has a CUDA devel image with `nvcc`, only `make` is needed.

## Run

By default the miner auto-detects all CUDA GPUs and starts one worker per GPU:

```bash
./midstate-cuda-miner \
  -o stratum+tcp://127.0.0.1:3333 \
  -a <YOUR_MSS_ADDRESS> \
  -w rig
```

Workers are suffixed automatically: `rig-gpu0`, `rig-gpu1`, and so on.
Auto mode also partitions nonce space across GPUs so cards do not duplicate
each other's shares on the same pool job.
Auto mode shows a live terminal dashboard with current and average hashrate,
accepted/rejected shares, submitted candidates, active job, and per-GPU status.

For plain line logs:

```bash
./midstate-cuda-miner -o stratum+tcp://127.0.0.1:3333 -a <YOUR_MSS_ADDRESS> -w rig --no-dashboard
```

To pin one GPU:

```bash
./midstate-cuda-miner \
  -o stratum+tcp://127.0.0.1:3333 \
  -a <YOUR_MSS_ADDRESS> \
  -w rig \
  -d 0
```

For an external miner connecting to your Clore pool mapping:

```bash
./midstate-cuda-miner \
  -o stratum+tcp://n1.us.clorecloud.net:1820 \
  -a <YOUR_MSS_ADDRESS> \
  -w rig
```

## Tuning

Defaults are conservative for Pascal:

```text
--blocks 4096
--threads 128
--batch 524288
```

Try:

```bash
./midstate-cuda-miner -o stratum+tcp://127.0.0.1:3333 -a <MSS> -w test --blocks 8192 --threads 128
./midstate-cuda-miner -o stratum+tcp://127.0.0.1:3333 -a <MSS> -w test --blocks 8192 --threads 256
```

Higher hashrate is good only if accepted shares also increase.
