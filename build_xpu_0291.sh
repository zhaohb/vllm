#!/usr/bin/env bash
IMG=vllm-xpu:0.29.1rc1
PROXY=http://proxy-ir.intel.com:912
NOPROXY=.sh.intel.com,localhost,127.0.0.0/8,10.239.0.0/16
docker build -f docker/Dockerfile.xpu -t $IMG \
    --build-arg GIT_REPO_CHECK=0 \
    --build-arg http_proxy=$PROXY --build-arg https_proxy=$PROXY --build-arg no_proxy=$NOPROXY \
    --build-arg HTTP_PROXY=$PROXY --build-arg HTTPS_PROXY=$PROXY --build-arg NO_PROXY=$NOPROXY \
    . 
echo "== built. verify 0.29.1rc1 xpu files + torch baked in image =="
docker run --rm --entrypoint bash $IMG -lc \
  'python3 -c \"import torch;print(\\\"torch\\\",torch.__version__)\"; \
   python3 -c \"import vllm,os;p=os.path.dirname(vllm.__file__);import glob;fs=glob.glob(p+\\\"/models/deepseek_v4/xpu/*.py\\\");print(\\\"dsv4 xpu files:\\\",len(fs));assert any(\\\"model.py\\\" in f for f in fs)\"'
echo "== build_xpu_main done: $IMG on $BUILD_NODE =="
