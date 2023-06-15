![funding logo](https://raw.githubusercontent.com/RECeSS-EU-Project/RECeSS-EU-Project.github.io/main/assets/images/header%2BEU_rescale.jpg)

# BENCHmark for drug Screening with COllaborative FIltering (benchscofi) Python Package

This repository is a part of the EU-funded [RECeSS project](https://recess-eu-project.github.io) (#101102016), and hosts the implementations and / or wrappers to published implementations of collaborative filtering-based algorithms for easy benchmarking.

## Install the latest release

### Using pip (package hosted on PyPI)

```bash
pip install benchscofi
```

### Using conda (package hosted on Anaconda.org)

```bash
conda install -c recess benchscofi
```

## Running the notebooks

## Example usage

Once installed, to import **benchscofi** into your Python code

```
import benchscofi
```

## Add a novel implementation / algorithm

Add a new Python file (extension .py) in ``src/benchscofi/`` named ``<model>`` (where ``model`` is the name of the algorithm), which contains a subclass of ``stanscofi.models.BasicModel`` **which has the same name as your Python file**. At least implement methods ``preprocessing``, ``fit``, ``model_predict``, and a default set of parameters (which is used for testing purposes). Please have a look at the placeholder file ``Constant.py`` which implements a classification algorithm which labels all datapoints as positive. 

It is highly recommended to provide a proper documentation of your class, along with its methods.

## Measure environmental impact

To mesure your environmental impact when using this package (in terms of carbon emissions), please run the following command

```
! codecarbon init
```

 to initialize the CodeCarbon config. For more information about using CodeCarbon, please refer to the [official repository](https://github.com/mlco2/codecarbon).

### Environment

It is strongly advised to create a virtual environment using Conda (python>=3.8)

```
conda create -n benchscofi_env python=3.8.5 -y
conda activate benchscofi_env
python3 -m pip install benchscofi ## or use the conda command above
python3 -m pip install notebook>=6.5.4 markupsafe==2.0.1 ## packages for Jupyter notebook
conda deactivate
conda activate benchscofi_env
jupyter notebook
```

The complete list of dependencies for *benchscofi* can be found at [requirements.txt](https://raw.githubusercontent.com/RECeSS-EU-Project/benchscofi/master/pip/requirements.txt) (pip) or [meta.yaml](https://raw.githubusercontent.com/RECeSS-EU-Project/benchscofi/master/conda/meta.yaml) (conda).

## Licence

This repository is under an [OSI-approved](https://opensource.org/licenses/) [MIT license](https://raw.githubusercontent.com/RECeSS-EU-Project/benchscofi/master/LICENSE). 

## Community guidelines with respect to contributions, issue reporting, and support

[Pull requests](https://github.com/RECeSS-EU-Project/benchscofi/pulls) and [issue flagging](https://github.com/RECeSS-EU-Project/benchscofi/issues) are welcome, and can be made through the GitHub interface. Support can be provided by reaching out to ``recess-project[at]proton.me``. However, please note that contributors and users must abide by the [Code of Conduct](https://github.com/RECeSS-EU-Project/benchscofi/blob/master/CODE%20OF%20CONDUCT.md).

