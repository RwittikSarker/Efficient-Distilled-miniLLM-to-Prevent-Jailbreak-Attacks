## Installation (Windows, CUDA 12.1)

> Requires an NVIDIA GPU and a driver compatible with CUDA 12.1.
> No CUDA Toolkit installation is required.

```bash
python -m virtualenv jailbreak
jailbreak\Scripts\activate

pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121
