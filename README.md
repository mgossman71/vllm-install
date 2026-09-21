# vLLM + Qwen3.8-27B on Ubuntu with an NVIDIA RTX 5090

This guide documents the **working installation path** for serving Qwen3.8-27B on a single NVIDIA RTX 5090 under Ubuntu using vLLM.

It intentionally excludes the failed/dead-end approaches encountered during setup, including:

- Using the Q4_K_M GGUF directly with vLLM.
- Installing Ubuntu's `nvidia-cuda-toolkit` package, which provided CUDA 12.0.
- Mixing FlashInfer package versions.
- Creating a `/usr/local/bin/nvcc` symlink.
- Running the model without the Qwen reasoning parser.

The target configuration is:

- Ubuntu 24.04
- NVIDIA RTX 5090, 32 GB VRAM
- NVIDIA driver exposed to the Ubuntu system/LXC
- CUDA 13.0 toolkit
- Python 3.12
- vLLM
- Qwen3.8-27B GPTQ INT4
- FP8 KV cache
- 180224 token maximum context
- OpenAI-compatible API
- Qwen reasoning separated from normal response content
- Cline-compatible automatic tool calling and Plan Mode

---

## 1. Proxmox LXC GPU passthrough

If this Ubuntu instance is a Proxmox LXC, expose the NVIDIA devices to the container.

Replace `136` with the actual LXC ID.

Run these commands **on the Proxmox host**:

```bash
pct set 136 -dev0 /dev/nvidia0
pct set 136 -dev1 /dev/nvidiactl
pct set 136 -dev2 /dev/nvidia-uvm
pct set 136 -dev3 /dev/nvidia-uvm-tools
pct set 136 -dev4 /dev/nvidia-caps/nvidia-cap1
pct set 136 -dev5 /dev/nvidia-caps/nvidia-cap2

pct stop 136
pct start 136
```

Inside the Ubuntu LXC, verify that the GPU is visible:

```bash
nvidia-smi
```

The output should identify:

```text
NVIDIA GeForce RTX 5090
```

> Important: If the RTX 5090 is passed through to multiple LXCs, they all share the same physical VRAM. Stop/unload models in other containers before starting vLLM.

---

## 2. Install base packages

Inside the Ubuntu server/LXC:

```bash
apt update
apt install -y \
  curl \
  wget \
  ca-certificates \
  gnupg \
  python3 \
  python3-pip
```

Ubuntu 24.04 normally provides Python 3.12.

Verify:

```bash
python3 --version
```

---

## 3. Install NVIDIA CUDA 13.0 toolkit

Do **not** install Ubuntu's `nvidia-cuda-toolkit` package. On Ubuntu 24.04 it can install an older CUDA toolkit that is not appropriate for this RTX 5090/vLLM configuration.

Install NVIDIA's CUDA repository:

```bash
cd /tmp

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb

dpkg -i cuda-keyring_1.1-1_all.deb

apt update
apt install -y cuda-toolkit-13-0
```

Set CUDA 13.0 globally:

```bash
cat >/etc/profile.d/cuda.sh <<'EOF'
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
EOF

chmod 644 /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh
hash -r
```

Verify:

```bash
which nvcc
nvcc --version
```

Expected path:

```text
/usr/local/cuda-13.0/bin/nvcc
```

Expected CUDA release:

```text
Cuda compilation tools, release 13.0
```

Also verify the CUDA runtime headers exist:

```bash
ls -l /usr/local/cuda-13.0/targets/x86_64-linux/include/cuda_runtime.h
```

### CUDA compiler sanity test

Before installing/running FlashInfer, verify that CUDA code can actually compile:

```bash
cat >/tmp/cudatest.cu <<'EOF'
#include <cuda_runtime.h>
int main() { return 0; }
EOF

nvcc /tmp/cudatest.cu -o /tmp/cudatest
/tmp/cudatest
echo $?
```

The final result should be:

```text
0
```

Do not proceed until this succeeds.

---

## 4. Install `uv`

Install Astral `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

Verify:

```bash
uv --version
```

---

## 5. Create the vLLM environment

Use `/opt/vllm` for the installation:

```bash
mkdir -p /opt/vllm
cd /opt/vllm

uv venv --python 3.12
source .venv/bin/activate
```

The shell prompt should now show the `vllm` virtual environment.

---

## 6. Install vLLM for CUDA 13.0

Install vLLM with the CUDA 13.0 PyTorch backend:

```bash
cd /opt/vllm
source .venv/bin/activate

uv pip install vllm --torch-backend=cu130
```

Verify the RTX 5090 and CUDA environment:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print(
        "VRAM GB:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    )
PY
```

A working RTX 5090 configuration should report values similar to:

```text
PyTorch: 2.x.x+cu130
CUDA build: 13.0
CUDA available: True
GPU: NVIDIA GeForce RTX 5090
Compute capability: (12, 0)
VRAM GB: ~31.36
```

Verify vLLM:

```bash
vllm --version
```

---

## 7. Install FlashInfer correctly

vLLM uses FlashInfer kernels on the RTX 5090.

The critical requirement is that these three packages use the **same FlashInfer version**:

- `flashinfer-python`
- `flashinfer-cubin`
- `flashinfer-jit-cache`

Install/upgrade the CUDA 13 FlashInfer Python package:

```bash
cd /opt/vllm
source .venv/bin/activate

uv pip install --upgrade "flashinfer-python[cu13]" \
  --index-url https://flashinfer.ai/whl
```

Read the version without importing FlashInfer:

```bash
FLASHINFER_VERSION="$(python - <<'PY'
import importlib.metadata
print(importlib.metadata.version("flashinfer-python"))
PY
)"

echo "$FLASHINFER_VERSION"
```

Install the **exact same version** of the cubin package:

```bash
uv pip install --upgrade \
  "flashinfer-cubin==${FLASHINFER_VERSION}" \
  --index-url https://flashinfer.ai/whl
```

Install the CUDA 13.0 JIT cache using the same FlashInfer version:

```bash
uv pip install --upgrade \
  "flashinfer-jit-cache==${FLASHINFER_VERSION}+cu130" \
  --index-url https://flashinfer.ai/whl/cu130
```

### Verify FlashInfer

Make sure CUDA is still in the active shell:

```bash
source /etc/profile.d/cuda.sh
```

Then:

```bash
flashinfer clear-cache
flashinfer show-config
```

The output should show:

```text
CUDA_VERSION: 13.0
CUDA_HOME: /usr/local/cuda-13.0
NVCC found: Yes
```

For the RTX 5090, it should identify the Blackwell architecture as SM120 / `(12, '0f')`.

The final module status should look like:

```text
Registered ... modules
compiled: ...
Not compiled: 0
```

Also verify the import:

```bash
python - <<'PY'
import flashinfer
print("FlashInfer:", flashinfer.__version__)
PY
```

### Important: do not create an `nvcc` symlink

Do **not** create:

```text
/usr/local/bin/nvcc -> /usr/local/cuda-13.0/bin/nvcc
```

Use the real CUDA location through `PATH` and `CUDA_HOME`.

Calling `nvcc` through an inappropriate symlink can interfere with CUDA's internal header/toolkit path discovery.

---

## 8. Model choice

The Ollama `qwen3.8:27b` setup uses a roughly 4-bit weight quantization. vLLM's GGUF support did not work for this Qwen3.8 model because the current GGUF adapter did not recognize the underlying `qwen3_5` model type.

For vLLM, use this GPTQ INT4 model instead:

```text
bernhardbrieger/Qwen3.8-27B-GPTQ-Int4
```

This gives the desired 4-bit-class model footprint while working with vLLM's optimized GPTQ/Marlin path.

The checkpoint is approximately 19.5 GiB.

---

## 9. Start Qwen3.8-27B

Before starting the model, make sure another Ollama/vLLM process is not already consuming the RTX 5090.

Check:

```bash
nvidia-smi
```

If another container is using the card, unload/stop that model first.

Start vLLM:

```bash
cd /opt/vllm
source .venv/bin/activate
source /etc/profile.d/cuda.sh

vllm serve bernhardbrieger/Qwen3.8-27B-GPTQ-Int4 \
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
  --tool-call-parser qwen3_coder
```

### What these options do

- `--host 0.0.0.0`
  Makes the API available to other systems on the network.

- `--port 8000`
  Runs the API on TCP port 8000.

- `--quantization gptq`
  Uses the GPTQ INT4 quantized model.

- `--max-model-len 180224`
  Configures the desired 180,224-token maximum context.

- `--gpu-memory-utilization 0.97`
  Allows vLLM to use up to 97% of the available RTX 5090 VRAM.

- `--kv-cache-dtype fp8`
  Stores the KV cache in FP8 to substantially reduce VRAM usage.

- `--max-num-seqs 1`
  Optimizes the server for one large active sequence rather than many concurrent clients.

- `--enable-prefix-caching`
  Allows repeated prompt prefixes to reuse cached KV data.

- `--reasoning-parser qwen3`
  Parses Qwen `<think>...</think>` output so reasoning is separated from normal response content.

- `--enable-auto-tool-choice`
  Allows OpenAI-compatible clients such as Cline to send `tool_choice: "auto"` and lets the model decide when to invoke tools.

- `--tool-call-parser qwen3_coder`
  Uses vLLM's Qwen3 Coder tool-call parser for Qwen3.8 tool output. This is the working parser for this model on vLLM 0.29.0 and is required for Cline's agent/tool workflow when Cline sends `tool_choice: "auto"`.

A successful startup ends with lines similar to:

```text
Starting vLLM server on http://0.0.0.0:8000
Application startup complete.
```

---

## 10. Test the server

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### List models

```bash
curl http://127.0.0.1:8000/v1/models
```

### Chat completion test

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bernhardbrieger/Qwen3.8-27B-GPTQ-Int4",
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ]
  }'
```

A correct response should contain normal output in:

```json
"content": "Hello!"
```

and Qwen's internal reasoning separately in:

```json
"reasoning": "..."
```

The API should also report reasoning-token usage separately.

---

## 11. Client configuration

For OpenAI-compatible clients such as Hermes or Cline, use:

```text
Base URL:
http://<LXC-IP>:8000/v1
```

Model:

```text
bernhardbrieger/Qwen3.8-27B-GPTQ-Int4
```

If the client requires an API key even though the local server does not enforce one, use a harmless placeholder such as:

```text
local
```

### Cline configuration

In Cline, select **OpenAI Compatible** and configure:

```text
Provider:           OpenAI Compatible
Base URL:           http://<LXC-IP>:8000/v1
API Key:            local
Model ID:           bernhardbrieger/Qwen3.8-27B-GPTQ-Int4
Context Window:     180224
Max Output Tokens:  16384
Compact Prompt:     Enabled
```

For the current server used while validating this guide:

```text
http://10.0.49.188:8000/v1
```

### Cline Plan Mode

Cline Plan Mode depends on reliable structured tool calling. With this Qwen3.8 model and vLLM 0.29.0, the working parser combination is:

```text
--reasoning-parser qwen3
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
```

A previous configuration using `--tool-call-parser hermes` allowed basic tool calling, but Cline Plan Mode could stop after a single reasoning response instead of continuing to inspect the project.

Using `qwen3_coder` corrected that behavior in testing.


Cline sends OpenAI-style tools and may send:

```json
"tool_choice": "auto"
```

Therefore the vLLM server **must** be started with:

```text
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
```

Without these flags, Cline can fail with:

```text
"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

Do **not** use:

```text
--tool-call-parser qwen3
```

on vLLM 0.29.0. `qwen3` is a valid **reasoning parser**, but it is not a registered tool-call parser in this release.

Use:

```text
--reasoning-parser qwen3
--tool-call-parser qwen3_coder
```

instead.

This parser combination also corrected Cline Plan Mode behavior where the model would begin reasoning about inspecting the workspace but stop before issuing file/tool calls.

---

## 12. Disk cleanup

vLLM, PyTorch, CUDA and model checkpoints consume substantial disk space.

### Inspect usage

```bash
df -h /

du -sh /root/.cache/* 2>/dev/null | sort -h
du -sh /root/.cache/huggingface/hub/* 2>/dev/null | sort -h
du -sh /opt/vllm
```

### Hugging Face cache

Use the Hugging Face cache CLI instead of manually deleting the shared `blobs` directory.

Inspect cached repositories:

```bash
hf cache ls
```

Dry-run removal:

```bash
hf cache rm model/<ORG>/<MODEL> --dry-run
```

Remove an unwanted repository:

```bash
hf cache rm model/<ORG>/<MODEL> -y
```

Prune incomplete/unreferenced downloads:

```bash
hf cache prune --dry-run
hf cache prune -y
```

### Safe disposable caches

These may be regenerated:

```bash
rm -rf /root/.cache/uv
rm -rf /root/.cache/vllm
rm -rf /root/.cache/flashinfer
apt clean
```

Do **not** delete the Hugging Face model repository currently used by vLLM unless you are prepared to download it again.

---

## 13. Avoid these mistakes

### Do not install Ubuntu's old CUDA toolkit

Avoid:

```bash
apt install nvidia-cuda-toolkit
```

For this setup it installed CUDA 12.0, while the RTX 5090/vLLM stack was using CUDA 13.0.

Use NVIDIA's:

```text
cuda-toolkit-13-0
```

instead.

### Do not mix FlashInfer versions

This will fail:

```text
flashinfer-python: 0.6.18
flashinfer-cubin:  0.6.18.post1
```

All FlashInfer packages must use the matching release.

### Do not use the vLLM GGUF plugin for this Qwen3.8 setup

The GGUF attempt failed with:

```text
Unknown gguf model_type: qwen3_5
```

Use the GPTQ INT4 model documented above.

### Do not ignore occupied GPU memory

If vLLM reports very little free VRAM but `nvidia-smi` inside the LXC shows no processes, another LXC may be using the same passed-through GPU.

Check the Proxmox host or the other GPU-enabled container.

### Do not omit the Qwen reasoning parser

Without:

```text
--reasoning-parser qwen3
```

the model may place its `<think>` reasoning directly into the normal `content` field.

### Do not omit tool-calling support when using Cline

Cline uses OpenAI-compatible tool calls and can send:

```json
"tool_choice": "auto"
```

The server must include:

```text
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
```

Otherwise vLLM returns:

```text
"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

For vLLM 0.29.0, do **not** substitute:

```text
--tool-call-parser qwen3
```

That produces an error similar to:

```text
KeyError: 'invalid tool call parser: qwen3'
```

The working parser combination for this setup is:

```text
--reasoning-parser qwen3
--tool-call-parser qwen3_coder
```

If Cline Plan Mode produces one reasoning response such as "Let me explore the workspace first" and then stops without reading files or calling tools, verify that the server was launched with `qwen3_coder`.

---

## 14. Quick repeat-install command sequence

This section is a condensed checklist for a fresh Ubuntu 24.04 system after NVIDIA GPU device passthrough is already working.

```bash
# Base packages
apt update
apt install -y curl wget ca-certificates gnupg python3 python3-pip

# NVIDIA CUDA repository + CUDA 13.0 toolkit
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
apt install -y cuda-toolkit-13-0

# Persistent CUDA environment
cat >/etc/profile.d/cuda.sh <<'EOF'
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
EOF
chmod 644 /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# vLLM environment
mkdir -p /opt/vllm
cd /opt/vllm
uv venv --python 3.12
source .venv/bin/activate

# vLLM + PyTorch CUDA 13.0
uv pip install vllm --torch-backend=cu130

# FlashInfer core for CUDA 13
uv pip install --upgrade "flashinfer-python[cu13]" \
  --index-url https://flashinfer.ai/whl

# Determine exact FlashInfer version
FLASHINFER_VERSION="$(python - <<'PY'
import importlib.metadata
print(importlib.metadata.version("flashinfer-python"))
PY
)"

# Matching FlashInfer kernel packages
uv pip install --upgrade \
  "flashinfer-cubin==${FLASHINFER_VERSION}" \
  --index-url https://flashinfer.ai/whl

uv pip install --upgrade \
  "flashinfer-jit-cache==${FLASHINFER_VERSION}+cu130" \
  --index-url https://flashinfer.ai/whl/cu130

# Verify
source /etc/profile.d/cuda.sh

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
PY

flashinfer clear-cache
flashinfer show-config
```

Then start the model:

```bash
cd /opt/vllm
source .venv/bin/activate
source /etc/profile.d/cuda.sh

vllm serve bernhardbrieger/Qwen3.8-27B-GPTQ-Int4 \
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
```

---

## 15. Final working configuration

```text
GPU:                NVIDIA GeForce RTX 5090
GPU architecture:   Blackwell / SM120
VRAM:               ~31.36 GiB usable
OS:                 Ubuntu 24.04
Python:             3.12
CUDA toolkit:       13.0
PyTorch backend:    cu130
Inference server:   vLLM
Model:              bernhardbrieger/Qwen3.8-27B-GPTQ-Int4
Weight quantization: GPTQ INT4
KV cache:           FP8
Context:            180224
Concurrency target: 1 sequence
API:                OpenAI-compatible
API port:           8000
Reasoning parser:   qwen3
Auto tool choice:   enabled
Tool-call parser:   qwen3_coder
Cline context:      180224
```

This is the configuration that successfully started the vLLM OpenAI-compatible server, returned a working Qwen3.8 chat-completion response on the RTX 5090, separated Qwen reasoning correctly, and supported Cline automatic tool calling.
