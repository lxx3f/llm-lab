# LLM Lab 本地环境

当前项目使用仓库内 Conda 环境：

```text
.venv/
```

环境基于已验证支持 RTX 5070 Ti `sm_120` 的 PyTorch 配置创建。

## 环境信息

- Python 3.12.13
- PyTorch 2.10.0+cu128
- torchvision 0.25.0+cu128
- PyYAML 6.0.3
- jsonschema 4.26.0
- CUDA runtime 12.8
- cuDNN 9.10.2
- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU
- Compute capability：`sm_120`（`(12, 0)`）
- 设备支持架构：`sm_70 / sm_75 / sm_80 / sm_86 / sm_90 / sm_100 / sm_120`

## 运行时快照

```text
torch: 2.10.0+cu128
cuda runtime: 12.8
cudnn: 91002
arch list: ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
device: NVIDIA GeForce RTX 5070 Ti Laptop GPU
compute cap: (12, 0)
jsonschema: 4.26.0
yaml: 6.0.3
```

复现命令：

```bash
.venv/Scripts/python.exe -c "import torch, jsonschema, yaml; \
  print('torch:', torch.__version__); \
  print('cuda runtime:', torch.version.cuda); \
  print('cudnn:', torch.backends.cudnn.version()); \
  print('arch list:', torch.cuda.get_arch_list()); \
  print('device:', torch.cuda.get_device_name(0)); \
  print('compute cap:', torch.cuda.get_device_capability(0)); \
  print('jsonschema:', jsonschema.__version__); \
  print('yaml:', yaml.__version__)"
```

## 使用方式

推荐直接使用环境中的解释器，避免依赖当前 shell 是否已激活环境：

```bash
.venv/Scripts/python.exe scripts/validate_stage0.py --examples
.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py" -v
.venv/Scripts/python.exe -m unittest discover -s architecture_lab/tests -p "test_*.py" -v
.venv/Scripts/python.exe scripts/run_dense_baseline.py
```

Windows PowerShell 激活：

```powershell
conda activate C:\Users\23236\repositories\llm-lab\.venv
```

Git Bash 激活：

```bash
source .venv/Scripts/activate
```

## GPU 验证

```bash
.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
```

应看到：

```text
True
NVIDIA GeForce RTX 5070 Ti Laptop GPU
...
'sm_120'
```

## 说明

`.venv` 是本地运行环境，不应提交到版本库。若项目启用 Git，应将 `.venv/` 加入 `.gitignore`。
