ENVS=$(conda env list | awk '{print $1}')

# if [[ $ENVS = *"$1"* ]]; then
#     echo "[PT INFO] \"$1\" already exists. Pass the installation"
# else 
#     echo "[PT INFO] Creating $1..."
#     conda create -n $1 python=3.12 -y
    conda activate "$1"
    echo "[PT INFO] Done !"

    echo "[PT INFO] Dependecies..."
    conda install nvidia/label/cuda-11.8.0::cuda -y
    conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -c conda-forge -y
    conda install -c anaconda h5py pyyaml -y
    conda install -c conda-forge sharedarray tensorboardx -y
    echo "[PT INFO] Done !"

    echo "[PT INFO] Installing cuda operations..."
    cd lib/pointops
    python3 setup.py install
    cd ../..
    echo "[PT INFO] Done !"

    NVCC="$(nvcc --version)"
    TORCH="$(python -c "import torch; print(torch.__version__)")"

    echo "[PT INFO] Finished the installation!"
    echo "[PT INFO] ========== Configurations =========="
    echo "$NVCC"
    echo "[PT INFO] PyTorch version: $TORCH"
    echo "[PT INFO] ===================================="

# fi;