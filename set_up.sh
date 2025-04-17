set -e

echo "setting LLM env: (Python 3.10 + CUDA 12.3 + L4 GPU)..."

sudo apt update && sudo apt upgrade -y
sudo apt install -y git wget curl build-essential libaio-dev libopenmpi-dev unzip

# install Miniconda
if ! command -v conda &> /dev/null; then
  echo "🔧 installing Miniconda..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
  bash ~/miniconda.sh -b -p $HOME/miniconda
  echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
  source ~/.bashrc
fi


source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
echo "✅ activated base env"

echo "🚀 installing PyTorch (cu123)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu123

echo "📦 installing Transformers、datasets、wandb..."
pip install \
  numpy \
  tqdm \
  transformers \
  datasets \
  wandb \
  huggingface_hub \
  accelerate \
  peft \
  bitsandbytes \
  protobuf \
  scikit-learn \
  sentencepiece \
  ninja \
  packaging \
  pandas \
  tensorboard \
  torch_tb_profiler \
  safetensors 

pip install deepspeed

echo "conda activate llm-env" >> ~/.bashrc

echo "✅ done"
