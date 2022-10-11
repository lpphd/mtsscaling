# An Empirical Evaluation of Multivariate Time Series Classification with Input Transformation across Different Dimensions

This repository contains the official implementation of the experiments described in "An Empirical Evaluation of
Multivariate Time Series Classification with Input Transformation across Different Dimensions", accepted at ICMLA 2022 .

## Requirements

The code is written in Python 3.7.11 and has the following main dependencies:

* numba==0.55.1
* sympy==1.10.1
* scipy==1.4.1
* sktime==0.10.1
* sktime-dl==0.2.0
* numpy==1.21.5
* scikit-learn==1.0.2
* pandas==1.3.5

The version of the [mpi4py](https://mpi4py.readthedocs.io/en/stable/index.html) library is 4.0.0.dev0 and has been
installed from its [repository](https://mpi4py.readthedocs.io/en/stable/install.html).

## Notes

For the WEASEL+MUSE, ResNet and InceptionTime models, MPI is used to distribute the workload of the experiments across
different nodes, but no communication among nodes is necessary.

There are no separate files for the baseline experiments, but these can be derived from the existing files by skipping
the data scaling alltogether.

## Datasets

The models are evaluated on a subset of
the [UEA multivariate dataset collection](https://www.timeseriesclassification.com/dataset.php).

## Results

The experiment metrics can be found under [Results](Results/), in the format [model\_name]
\_uea\_metrics\_[scaling\_method]
\_[dimension].csv

The baseline metrics are in the format [model\_name]\_uea\_metrics\_none.csv
