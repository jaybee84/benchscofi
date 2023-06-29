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
