```bash
vi /etc/systemd/system/vllm.service
```
```bash
[Unit]
Description=vLLM Qwen3.8-27B API Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/vllm

Environment="CUDA_HOME=/usr/local/cuda-13.0"
Environment="PATH=/usr/local/cuda-13.0/bin:/opt/vllm/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/targets/x86_64-linux/lib"

ExecStart=/opt/vllm/.venv/bin/vllm serve bernhardbrieger/Qwen3.8-27B-GPTQ-Int4 \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization gptq \
  --max-model-len 180224 \
  --gpu-memory-utilization 0.97 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 1 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --kv-cache-metrics \
  --kv-cache-metrics-sample 0.01 \
  --tool-call-parser qwen3_coder

Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```
