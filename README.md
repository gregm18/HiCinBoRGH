Welcome to 
# HiCinBoRGH [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

by Greg Mercader and Dr. Benjamin Soibam

HiCinBoRGH (Hi-C in Blocks of Refined Groups of Hessian) is a chromatin loop detector utilizing scale-space theory, aiming to take the best concepts from scale-space theory and unsupervised machine learning methods and put them in a powerful, user-friendly approach to chromatin loop detection. It is currently heavily based on the program, MUSTACHE, for its innovative concepts on the subject.

Despite the name, Hessian, being used, users are given the option to use Difference of Gaussian and Determinant of Hessian accordingly and developers are free to add more types of scale-space representation as technology advances.



Though HiCinBoRGH can [currently] be ran on Windows operating systems, it is best to run in WSL/Ubuntu or Linux, due to dependencies potentially struggling on Windows operating systems.


## Installation and Running HiCinBoRGH

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

## How to use HiCinBoRGH

Before you run HiCinBoRGH, you must do the following:
1. Place your .mcool file in the data folder
2. If making comparisons to MUSTACHE, place MUSTACHE's output directly in the data folder alongside your .mcool file with the following name:
```bash
chr[chromosome number]_out[raw resolution divided by 1000]_pt[p-value threshold with no decimal].tsv
```
For example:
```bash
chr1_out10_pt005.tsv
```
This represents the result of chromosome 1 at 10kb resolution with a 0.05 p-value threshold retrieved from MUSTACHE. For accurate comparisons, it is expected that a sparsity threshold of 0.8 is used in the given MUSTACHE output.

Once this is complete and you run the program, HiCinBoRGH will ask you for a line of parameters that HiCinBoRGH will use for its chromatin loop detection process.

Here is the list of parameters in order of entry:
`<mcool_file> <start_chr> <end_chr> <resolution> <p_value> <norm> <dog/doh> <comparison to mustache>`

### Examples
Here are some examples of lines of parameters to run HiCinBoRGH:

Example 1: Running GSE63525_GM12878_diploid_maternal.mcool on chromosome 5 only at 5kb resolution with a 0.1 p-value threshold, Knight-Ruiz normalization, Determinant of Hessian scale-space representation, and comparison to MUSTACHE
```bash
GSE63525_GM12878_diploid_maternal.mcool 5 5 5kb 0.1 kr doh y
```

Example 2: Running GSE63525_GM12878_insitu_DpnII_combined_30.mcool on chromosomes 1 through 22 at 10kb resolution with a 0.05 p-value threshold, Knight-Ruiz normalization, Determinant of Hessian scale-space representation, and comparison to MUSTACHE
```bash
GSE63525_GM12878_insitu_DpnII_combined_30.mcool 1 22 10kb 0.05 kr doh y
```

Example 3: Running GSE63525_GM12878_insitu_DpnII_combined_30.mcool on chromosome X at 10kb resolution with a 0.05 p-value threshold, ICE normalization, Difference of Gaussian scale-space representation, and no comparison to MUSTACHE
```bash
GSE63525_GM12878_insitu_DpnII_combined_30.mcool 23 23 10kb 0.05 ice dog n
```

## Parameters

Here is the list of parameters in order of entry:

`<mcool_file> <start_chr> <end_chr> <resolution> <p_value> <norm> <dog/doh> <comparison to mustache>`

Here is a table describing each parameter for an in-depth understanding:

Parameter Name | Data Format | Description
--- | --- | ---
`<mcool_file>` | string | This is where the exact file name (including extension format) is written. e.g.: filename.mcool
`<start_chr>` | int (integer) | The first chromosome to be checked for chromatin loops. Chromosomes X, Y, and MT are 23, 24, and 25 respectively. e.g.: 7
`<end_chr>` | int (integer) | The last chromosome to be checked for chromatin loops. If you are only looking at one chromosome, simply make this number the same as your start_chr number. Chromosomes X, Y, and MT are 23, 24, and 25 respectively. e.g.: 3, 7
`<resolution>` | int + kb/mb | This is the resolution HiCinBoRGH will open your chromosome(s) at. HiCinBoRGH will only be able to open the resolution if it is available in your .mcool file. e.g.: 25kb, 10kb, 5kb
`<p_value>` | float (decimal number) | The threshold of p-value in which loops are valuable according to Benjamini-Hochberg procedure.
`<norm>` | string | The normalization your file is opened at as read by cooler. e.g.: KR, ICE, NONE (simply requests the raw matrix)
`<dog/doh>` | string | The scale-space representation you wish HiCinBoRGH to detect chromatin loops with. e.g.: dog = Difference of Gaussian, doh = Determinant of Hessian.
`<comparison to mustache>` | string | Confirms whether you wish to compare your HiCinBoRGH output to your MUSTACHE output. When comparing, output from MUSTACHE must be placed in the data folder. e.g: y, n

## Input Format
Currently, HiCinBoRGH only has support for .mcool files. All .mcool files are to be placed in the data folder. If you are making comparisons to MUSTACHE output, then your .tsv files are to be placed in the data folder as well. 

If you have a .hic file, the hic2cool python package has a simple way of converting .hic files to .mcool files. The cooler dependency installed with this program also has the abiltiy to add balancing weights to your new .mcool file if necessary.

## Output Format
HiCinBoRGH produces a new .csv file that gives a list of every loop with the category, chromosome number, coordinates, sigma scale they were detected at, scale-space score, and p and q values.

Here is the set up as follows
`Category, Chromosome, bin1, bin2, DoH/DoG score, p-value, q`



