Welcome to 
# HiCinBoRGH [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

by Greg Mercader and Dr. Benjamin Soibam

HiCinBoRGH (Hi-C in Blocks of Refined Groups of Hessian) is a chromatin loop detector utilizing scale-space theory, aiming to take the best concepts from scale-space theory and unsupervised machine learning methods and put them in a powerful, user-friendly approach to chromatin loop detection. It is currently heavily based on the program, MUSTACHE, for its innovative concepts on the subject.

Despite the name, Hessian, being used, users are given the option to use Difference of Gaussian and Determinant of Hessian accordingly and developers are free to add more types of scale-space representation as technology advances.



Though HiCinBoRGH can [currently] be ran on Windows operating systems, it is best to run in WSL/Ubuntu or Linux, due to dependencies potentially struggling on Windows operating systems.


## Installation

Here are the following requirements for each installation method:
1. GIT
2. Python version 3.11

### Installing with only Python on Linux or WSL/Ubuntu

When installing without external methods, simply open your terminal or Windows Powershell then power on WSL/Ubuntu and run the following commands.
```bash
git clone https://github.com/gregm18/HiCinBoRGH.git
python3 -m venv HiCinBoRGH/.venv
HiCinBoRGH/.venv/bin/python -m pip install HiCinBoRGH
```

When the program is installed, proceed to running the program.

#### Running

```bash
source HiCinBoRGH/.venv/bin/activate
python3 HiCinBoRGH/code/hicinborgh.py
```


### Installing with only Python on Windows

When installing without external methods, simply open Windows Powershell run the following lines of commands.

```bash
git clone https://github.com/gregm18/HiCinBoRGH.git
python -m venv HiCinBoRGH\.venv
HiCinBoRGH\.venv\Scripts\python.exe -m pip install HiCinBoRGH
```

When the program is installed, proceed to running the program.

#### Running

```bash
HiCinBoRGH\.venv\Scripts\Activate.ps1
python HiCinBoRGH\code\hicinborgh.py
```

### Installing using Conda

With Conda installed, simply run the following lines of commands in a terminal using WSL/Linux to install the program. 
This will install the program and create an environment named "hicinborgh"

```bash
git clone https://github.com/gregm18/HiCinBoRGH.git
conda env create -f ./mustache/environment.yml
```

#### Running

The following lines of commands will activate the hicinborgh environment and run the program via the hicinborgh.py file.

```bash
conda activate hicinborgh
python3 HiCinBoRGH/code/hicinborgh.py
```

## Dependencies
Here are the list of dependencies that HiCinBoRGH relies on:
  - python=3.11
  - numpy
  - scikit-image=0.26.0
  - cooler=0.10.4
  - scipy
  - statsmodels
  - psutil
  - scikit-learn
  - matplotlib


