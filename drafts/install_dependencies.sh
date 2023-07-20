#!/bin/bash

python3 -m venv cfdr
source cfdr/bin/activate
pip install -r requirements.txt
export PATH=$PATH:~/.local/bin
wd=$(pwd)
cd ~
git clone https://github.com/alexeygrigorev/libffm-python.git ffm
cd ffm
python3 setup.py install
cd ~
#https://elliot.readthedocs.io/en/v0.2.1/guide/alg_intro.html
git clone https://github.com//sisinflab/elliot.git
cd elliot
pip install --upgrade pip
pip install -e . --verbose
cd ~
git clone https://github.com/cjlin1/libmf.git
cd $wd


#apt-get install nvidia-smi
#nvidia-smi --list-gpus | wc -l # check the number of GPUs
#apt-get install linux-headers-$(uname -r)
#https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_local
#PACKAGE_NAME="cuda-repo-debian11-12-1-local_12.1.1-530.30.02-1_amd64.deb"
#wget "https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/"$PACKAGE_NAME
#sudo dpkg -i $PACKAGE_NAME
#sudo cp /var/cuda-repo-debian11-12-1-local/cuda-D8654E34-keyring.gpg /usr/share/keyrings/
#MDSUM=$(md5sum $PACKAGE_NAME | cut -d" " -f1)
#wget -O md5sum_cuda.txt https://developer.download.nvidia.com/compute/cuda/12.1.1/docs/sidebar/md5sum.txt
#TRUE_MDSUM=$(cat md5sum_cuda.txt | grep $PACKAGE_NAME | cut -d" " -f1)
#if [ "$TRUE_MDSUM" == "$MDSUM" ]; then echo "OK"; else echo "not OK"; fi

lspci | grep -i nvidia

#https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
conda activate stanscofi_env
lspci | grep ' VGA ' | cut -d" " -f 1 | xargs -i lspci -v -s {}
#conda uninstall cudatoolkit -y
apt-get install nvidia-cudnn -y
conda install cudnn -y
find / -name 'libcudnn'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kali/miniconda3/envs/stanscofi_env/lib
## Test
## https://gist.github.com/Quasimondo/7e1068e488e20f194d37ba80696b55d8
python3 -c "from tensorflow.python.client import device_lib;device_lib.list_local_devices();import tensorflow as tf;tf.config.list_physical_devices('GPU');print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"