![funding logo](https://raw.githubusercontent.com/RECeSS-EU-Project/RECeSS-EU-Project.github.io/main/assets/images/header%2BEU_rescale.jpg)

# BENCHmark for drug Screening with COllaborative FIltering (benchscofi) Python Package

This repository is a part of the EU-funded [RECeSS project](https://recess-eu-project.github.io) (#101102016), and hosts the implementations and / or wrappers to published implementations of collaborative filtering-based algorithms for easy benchmarking.

## Benchmark AUC values (default parameters, single random training/testing set split) [updated 7/20/23]

These values (rounded to the closest 3rd decimal place) can be obtained by the following commands 

```bash
cd tests/ && python3 -m test_models <algorithm> <dataset or empty if using the synthetic dataset>
```

  Algorithm                | Synthetic*    | TRANSCRIPT    [a] | Gottlieb [b]  | Cdataset [c] | PREDICT    [d] | LRSSL [e] | 
-------------------------- | ------------- | ----------------- | ------------- | ------------ | -------------- | --------- |
PMF [1]                    |  0.974        |  0.549            |  0.561        |  0.555       |  0.568         | 0.546     |
PulearnWrapper [2]         |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
ALSWR [3]                  |  0.745        |  0.567            |  0.582        |  0.608       | 0.621          | 0.604     |
FastaiCollabWrapper [4]    |  0.503        |  0.639            |  0.516        |  0.562       | 0.501          | 0.514     |
SimplePULearning [5]       |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
SimpleBinaryClassifier [6] |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
NIMCGCN [7]                |  0.500        |  0.500            |  0.500        |  0.500       |  N/A           | N/A       |
FFMWrapper [8]             |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
VariationalWrapper [9]     |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
DRRS [10]                  |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
SCPMF [11]                 |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
BNNR [12]                  |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
LRSSL [13]                 |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
MBiRW [14]                 |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |

*Synthetic dataset created with function ``generate_dummy_dataset`` in ``stanscofi.datasets`` and the following arguments:
```python
npositive=200 #number of positive pairs
nnegative=100 #number of negative pairs
nfeatures=50 #number of pair features
mean=0.5 #mean for the distribution of positive pairs, resp. -mean for the negative pairs
std=1 #standard deviation for the distribution of positive and negative pairs
```

[a] N/A
[b] N/A
[c] N/A
[d] N/A
[e] N/A

[1] N/A
[2] N/A
[3] N/A
[4] N/A
[5] N/A
[6] N/A
[7] N/A 
[8] N/A
[9] N/A
[10] N/A
[11] N/A
[12] N/A
[13] N/A
[14] N/A

## Statement of need

## Installation

### 1. Dependencies

#### R

Or ignore the following algorithms: ...

#### MATLAB/Octave

Or ignore the following algorithms: ...

```bash
apt-get install -y octave
```

#### MATLAB compiler

Or ignore the following algorithms: ...

```bash
sudo apt-get install -y libxmu-dev # libXmu.so.6 is required
wget -O MCR_R2012b_glnxa64_installer.zip https://ssd.mathworks.com/supportfiles/MCR_Runtime/R2012b/MCR_R2012b_glnxa64_installer.zip
mv MCR_R2012b_glnxa64_installer.zip /tmp
cd /tmp
unzip MCR_R2012b_glnxa64_installer.zip -d MCRInstaller
cd MCRInstaller
mkdir -p /usr/local/MATLAB/MATLAB_Compiler_Runtime/v80
chown -R kali /usr/local/MATLAB/
./install -mode silent -agreeToLicense  yes
```

#### Others (Elliot, FFM, libFM)

Or ignore the following algorithms: ...

```bash
bash drafts/install_dependencies.sh
```

### 2. Install CUDA (for tensorflow and pytorch-based algorithms)

Or ignore the following algorithms: ...

Install [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

### 3. Install the latest release

Using ``pip`` (package hosted on PyPI) or ``conda`` (package hosted on Anaconda.org)


```bash
pip install benchscofi # using pip
conda install -c recess benchscofi # or conda
```

## Example usage

### 0. Environment

It is strongly advised to create a virtual environment using Conda (python>=3.8)

```bash
conda create -n benchscofi_env python=3.8.5 -y
conda activate benchscofi_env
python3 -m pip install benchscofi ## or use the conda command above
python3 -m pip uninstall werkzeug
python3 -m pip install notebook>=6.5.4 markupsafe==2.0.1 ## packages for Jupyter notebook
conda deactivate
conda activate benchscofi_env
jupyter notebook
```

The complete list of dependencies for *benchscofi* can be found at [requirements.txt](https://raw.githubusercontent.com/RECeSS-EU-Project/benchscofi/master/pip/requirements.txt) (pip) or [meta.yaml](https://raw.githubusercontent.com/RECeSS-EU-Project/benchscofi/master/conda/meta.yaml) (conda).

### 1. Import module

Once installed, to import **benchscofi** into your Python code

```python
import benchscofi
```

### 2. Run notebooks

- Check out notebook ``Class prior estimation.ipynb`` to see tests of the class prior estimation methods on synthetic and real-life datasets.

- Check out notebook ``RankingMetrics.ipynb`` for example of training and prediction, along with the definitions of ranking metrics present in **stanscofi**.

- ... the list of notebooks is growing!

### 3. Measure environmental impact

To mesure your environmental impact when using this package (in terms of carbon emissions), please run the following command

```bash
! codecarbon init
```

 to initialize the CodeCarbon config. For more information about using CodeCarbon, please refer to the [official repository](https://github.com/mlco2/codecarbon).

## Licence

This repository is under an [OSI-approved](https://opensource.org/licenses/) [MIT license](https://raw.githubusercontent.com/RECeSS-EU-Project/benchscofi/master/LICENSE). 

## Community guidelines with respect to contributions, issue reporting, and support

You are more than welcome to add your own algorithm to the package!

#### Add a novel implementation / algorithm

Add a new Python file (extension .py) in ``src/benchscofi/`` named ``<model>`` (where ``model`` is the name of the algorithm), which contains a subclass of ``stanscofi.models.BasicModel`` **which has the same name as your Python file**. At least implement methods ``preprocessing``, ``fit``, ``model_predict``, and a default set of parameters (which is used for testing purposes). Please have a look at the placeholder file ``Constant.py`` which implements a classification algorithm which labels all datapoints as positive. 

It is highly recommended to provide a proper documentation of your class, along with its methods.

#### Rules for contributors

[Pull requests](https://github.com/RECeSS-EU-Project/benchscofi/pulls) and [issue flagging](https://github.com/RECeSS-EU-Project/benchscofi/issues) are welcome, and can be made through the GitHub interface. Support can be provided by reaching out to ``recess-project[at]proton.me``. However, please note that contributors and users must abide by the [Code of Conduct](https://github.com/RECeSS-EU-Project/benchscofi/blob/master/CODE%20OF%20CONDUCT.md).
