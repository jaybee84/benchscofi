![funding logo](https://raw.githubusercontent.com/RECeSS-EU-Project/RECeSS-EU-Project.github.io/main/assets/images/header%2BEU_rescale.jpg)

# BENCHmark for drug Screening with COllaborative FIltering (benchscofi) Python Package

This repository is a part of the EU-funded [RECeSS project](https://recess-eu-project.github.io) (#101102016), and hosts the implementations and / or wrappers to published implementations of collaborative filtering-based algorithms for easy benchmarking.

## Benchmark average disease-wise and global AUC values (default parameters, single random training/testing set split) [updated 8/02/23]

These values (rounded to the closest 3rd decimal place) can be reproduced using the following command

```bash
cd tests/ && python3 -m test_models <algorithm> <dataset or empty if using the synthetic dataset>
```

:no_entry:'s represent failure to train or to predict. ``N/A``'s have not been tested yet.

  Algorithm  (row-wise)    | Synthetic*    | TRANSCRIPT    [a] | Gottlieb [b]  | Cdataset [c] | PREDICT    [d] | LRSSL [e] | 
-------------------------- | ------------- | ----------------- | ------------- | ------------ | -------------- | --------- |
PMF [1]                    |  0.974        |  0.549            |  0.561        |  0.555       |  0.568         | 0.546     |
PulearnWrapper [2]         |  0.500        |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
ALSWR [3]                  |  0.755        |  0.567            |  0.582        |  0.608       |  0.621         | 0.604     |
FastaiCollabWrapper [4]    |  0.500        |  0.493            |  0.500        |  0.500       |  0.501         | 0.500     |
SimplePULearning [5]       |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
SimpleBinaryClassifier [6] |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
NIMCGCN [7]                |  0.500        |  0.500            |  0.500        |  0.500       |  :no_entry:    | 0.500     |
FFMWrapper [8]             |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
VariationalWrapper [9]     |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
DRRS [10]                  |  :no_entry:   |  0.542            |  0.647        |  0.685       |  :no_entry:    | 0.752     |
SCPMF [11]                 |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
BNNR [12]                  |  0.500        |  0.500            |  0.500        |  0.500       |  :no_entry:    | 0.500     |
LRSSL [13]                 |  0.509        |  :no_entry:       |  0.505        |  0.500       |  :no_entry:    | 0.495     |
MBiRW [14]                 |  0.501        |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
LibMFWrapper [15]          |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
LogisticMF [16]            |  0.500        |  0.500            |  0.500        |  0.500       |  0.500         | 0.500     |

  Algorithm  (global)      | Synthetic*    | TRANSCRIPT    [a] | Gottlieb [b]  | Cdataset [c] | PREDICT    [d] | LRSSL [e] | 
-------------------------- | ------------- | ----------------- | ------------- | ------------ | -------------- | --------- |
PMF [1]                    |  0.922        |  0.579            |  0.598        |  0.604       |  0.656         | 0.611     |
PulearnWrapper [2]         |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
ALSWR [3]                  |  0.971        |  0.507            |  0.677        |  0.724       |  0.693         | 0.685     |
FastaiCollabWrapper [4]    |  1.000        |  0.876            |  0.856        |  0.837       |  N/A           | N/A       |
SimplePULearning [5]       |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
SimpleBinaryClassifier [6] |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
NIMCGCN [7]                |  N/A          |  N/A              |  N/A          |  N/A         |  :no_entry:    | N/A       |
FFMWrapper [8]             |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
VariationalWrapper [9]     |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
DRRS [10]                  |  :no_entry:   |  0.662            |  0.838        |  0.878       |  :no_entry:    | 0.892     |
SCPMF [11]                 |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
BNNR [12]                  |  1.000        |  0.922            |  0.949        |  0.959       |  :no_entry:    | 0.972     |
LRSSL [13]                 |  0.127        |  :no_entry:       |  0.159        |  0.846       |  :no_entry:    | 0.665     |
MBiRW [14]                 |  1.000        |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
LibMFWrapper [15]          |  N/A          |  N/A              |  N/A          |  N/A         |  N/A           | N/A       |
LogisticMF [16]            |  1.000        |  0.910            |  0.941        |  0.955       |  0.953         | 0.933     |

*Synthetic dataset created with function ``generate_dummy_dataset`` in ``stanscofi.datasets`` and the following arguments:
```python
npositive=200 #number of positive pairs
nnegative=100 #number of negative pairs
nfeatures=50 #number of pair features
mean=0.5 #mean for the distribution of positive pairs, resp. -mean for the negative pairs
std=1 #standard deviation for the distribution of positive and negative pairs
```

---

[a] Réda, Clémence. (2023). TRANSCRIPT drug repurposing dataset (2.0.0) [Data set]. Zenodo. doi:10.5281/zenodo.7982976

[b] Gottlieb, A., Stein, G. Y., Ruppin, E., & Sharan, R. (2011). PREDICT: a method for inferring novel drug indications with application to personalized medicine. Molecular systems biology, 7(1), 496.

[c] Luo, H., Li, M., Wang, S., Liu, Q., Li, Y., & Wang, J. (2018). Computational drug repositioning using low-rank matrix approximation and randomized algorithms. Bioinformatics, 34(11), 1904-1912.

[d] Réda, Clémence. (2023). PREDICT drug repurposing dataset (2.0.1) [Data set]. Zenodo. doi:10.5281/zenodo.7983090

[e] Liang, X., Zhang, P., Yan, L., Fu, Y., Peng, F., Qu, L., … & Chen, Z. (2017). LRSSL: predict and interpret drug–disease associations based on data integration using sparse subspace learning. Bioinformatics, 33(8), 1187-1196.

---

Tags are associated with each method. 

- [[featureless]] means that the algorithm does not leverage the input of drug/disease features. 

- [[matrix_input]] means that the algorithm considers as input a matrix of ratings (plus possibly matrices of drug/disease features), instead of considering as input (drug, disease) pairs.

[1] Probabilistic Matrix Factorization (using Bayesian Pairwise Ranking) implemented at [this page](https://ethen8181.github.io/machine-learning/recsys/4_bpr.html). [[featureless]] [[matrix_input]]

[2] Elkan and Noto's classifier based on SVMs (package [pulearn](https://pulearn.github.io/pulearn/) and [paper](https://cseweb.ucsd.edu/~elkan/posonly.pdf)). [[featureless]]

[3] Alternating Least Square Matrix Factorization algorithm implemented at [this page](https://ethen8181.github.io/machine-learning/recsys/2_implicit.html#Implementation). [[featureless]] 

[4] Collaborative filtering approach ``collab_learner`` implemented by package [fast.ai](https://docs.fast.ai/collab.html). [[featureless]]

[5] Customizable neural network architecture with positive-unlabeled risk.
 
[6] Customizable neural network architecture for positive-negative learning.

[7] Jin Li, Sai Zhang, Tao Liu, Chenxi Ning, Zhuoxuan Zhang and Wei Zhou. Neural inductive matrix completion with graph convolutional networks for miRNA-disease association prediction. Bioinformatics, Volume 36, Issue 8, 15 April 2020, Pages 2538–2546. doi: 10.1093/bioinformatics/btz965. ([implementation](https://github.com/ljatynu/NIMCGCN)).

[8] Field-aware Factorization Machine (package [pyFFM](https://pypi.org/project/pyFFM/)).

[9] Vie, J. J., Rigaux, T., & Kashima, H. (2022, December). Variational Factorization Machines for Preference Elicitation in Large-Scale Recommender Systems. In 2022 IEEE International Conference on Big Data (Big Data) (pp. 5607-5614). IEEE. ([pytorch implementation](https://github.com/jilljenn/vae)). [[featureless]] 

[10] Luo, H., Li, M., Wang, S., Liu, Q., Li, Y., & Wang, J. (2018). Computational drug repositioning using low-rank matrix approximation and randomized algorithms. Bioinformatics, 34(11), 1904-1912. ([download](http://bioinformatics.csu.edu.cn/resources/softs/DrugRepositioning/DRRS/index.html)). [[matrix_input]] 

[11] Meng, Y., Jin, M., Tang, X., & Xu, J. (2021). Drug repositioning based on similarity constrained probabilistic matrix factorization: COVID-19 as a case study. Applied soft computing, 103, 107135. ([implementation](https://github.com/luckymengmeng/SCPMF)). ??

[12] Yang, M., Luo, H., Li, Y., & Wang, J. (2019). Drug repositioning based on bounded nuclear norm regularization. Bioinformatics, 35(14), i455-i463. ([implementation](https://github.com/BioinformaticsCSU/BNNR)). [[matrix_input]] 

[13] Liang, X., Zhang, P., Yan, L., Fu, Y., Peng, F., Qu, L., ... & Chen, Z. (2017). LRSSL: predict and interpret drug–disease associations based on data integration using sparse subspace learning. Bioinformatics, 33(8), 1187-1196. ([implementation](https://github.com/LiangXujun/LRSSL)). [[matrix_input]] 

[14] Luo, H., Wang, J., Li, M., Luo, J., Peng, X., Wu, F. X., & Pan, Y. (2016). Drug repositioning based on comprehensive similarity measures and bi-random walk algorithm. Bioinformatics, 32(17), 2664-2671. ([implementation](https://github.com/bioinfomaticsCSU/MBiRW)). [[matrix_input]] 

[15] W.-S. Chin, B.-W. Yuan, M.-Y. Yang, Y. Zhuang, Y.-C. Juan, and C.-J. Lin. LIBMF: A Library for Parallel Matrix Factorization in Shared-memory Systems. JMLR, 2015. [[featureless]]

[16] Johnson, C. C. (2014). Logistic matrix factorization for implicit feedback data. Advances in Neural Information Processing Systems, 27(78), 1-9. [[featureless]]

---

## Statement of need

As of 2022, current drug development pipelines last around 10 years, costing $2billion in average, while drug commercialization failure rates go up to 90%. These issues can be mitigated by drug repurposing, where chemical compounds are screened for new therapeutic indications in a systematic fashion. In prior works, this approach has been implemented through collaborative filtering. This semi-supervised learning framework leverages known drug-disease matchings in order to recommend new ones.

There is no standard pipeline to train, validate and compare collaborative filtering-based repurposing methods, which considerably limits the impact of this research field. In **benchscofi**, the estimated improvement over the state-of-the-art (implemented in the package) can be measured through adequate and quantitative metrics tailored to the problem of drug repurposing across a large set of publicly available drug repurposing datasets.

## Installation

Platforms: Linux & Mac (developed and tested).

Python: 3.8.*

### 1. Dependencies

#### R

Install R based on your distribution, or do not use the following algorithms: ``LRSSL``. Check if R is properly installed using the following command

```bash
R -q -e "print('R is installed and running.')"
```

#### MATLAB / Octave

Install MATLAB or Octave (free) based on your distribution, or do not use the following algorithms: ``BNNR``, ``SCPMF``, ``MBiRW``. Check if Octave is properly installed using the following command

```bash
octave --eval "'octave is installed!'"
```

#### MATLAB compiler

Install a MATLAB compiler (version 2012b) as follow, or do not use algorithm ``DRRS``.

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

### 2. Install CUDA (for tensorflow and pytorch-based algorithms)

TODO

Or ignore the following algorithms: ...

Install [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

### 3. Install the latest **benchscofi** release

Using ``pip`` (package hosted on PyPI) or ``conda`` (package hosted on Anaconda.org)


```bash
pip install benchscofi # using pip
conda install -c recess benchscofi # or conda
```

## Example usage

Further documentation can be found at [the following page](https://recess-eu-project.github.io/benchscofi) [TODO].

### 0. Environment

It is strongly advised to create a virtual environment using Conda (python>=3.8)

```bash
conda create -n benchscofi_env python=3.8.5 -y
conda activate benchscofi_env
python3 -m pip install benchscofi ## or use the conda command above
## TODO python3 -m pip uninstall werkzeug
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

- Check out notebook ``RankingMetrics.ipynb`` for example of training with cross-validation and evaluation of the model predictions, along with the definitions of ranking metrics present in **stanscofi**. It also runs [*libmf*](https://github.com/cjlin1/libmf/).

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

### 1. Add a novel implementation / algorithm

Add a new Python file (extension .py) in ``src/benchscofi/`` named ``<model>`` (where ``model`` is the name of the algorithm), which contains a subclass of ``stanscofi.models.BasicModel`` **which has the same name as your Python file**. At least implement methods ``preprocessing``, ``fit``, ``model_predict``, and a default set of parameters (which is used for testing purposes). Please have a look at the placeholder file ``Constant.py`` which implements a classification algorithm which labels all datapoints as positive. 

It is highly recommended to provide a proper documentation of your class, along with its methods.

### 2. Rules for contributors

[Pull requests](https://github.com/RECeSS-EU-Project/benchscofi/pulls) and [issue flagging](https://github.com/RECeSS-EU-Project/benchscofi/issues) are welcome, and can be made through the GitHub interface. Support can be provided by reaching out to ``recess-project[at]proton.me``. However, please note that contributors and users must abide by the [Code of Conduct](https://github.com/RECeSS-EU-Project/benchscofi/blob/master/CODE%20OF%20CONDUCT.md).
