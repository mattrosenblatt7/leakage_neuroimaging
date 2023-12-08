This repository includes data and code for the preprint [*The effects of data leakage on neuroimaging predictive models*](https://www.biorxiv.org/content/10.1101/2023.06.09.544383v1)

The following code and data are for non-commercial and academic purposes only.

# Software requirements

This code requires Python 3. While it may work with various versions of Python and the following packages, the code was specifically developed and tested with Python 3.11.3 and the following packages:

* mat73 0.60
* matplotlib 3.8.0
* numpy 1.24.3
* pandas 2.0.3
* scikit-learn 1.2.2
* scipy 1.10.1
* seaborn 0.13.0
* tqdm 4.66.1

Beyond installing python and these packages, no specific installation is required. Installation of python and the packages should take about 10 minutes. To reproduce the plots without any installation required, please see the **Plotting** section. Please note that the analysis in this paper is likely not practical to run on a personal computer, due to the many simulations and computational resources involved. In addition, the data cannot be shared, but please see the **Datasets** section below for links to the unprocessed fMRI data (note: many require applications to access the data).

# Main leakage analysis

The main leakage analysis is found in the file [run_leakage.py](run_leakage.py)

You can input the type of leakage you want to perform, the number of cross-validation folds k, the percentage of features to use, the sample size you want, and the seed for resampling. Arguments are further detailed in the script, but an example call is below:

```
python leakage_sample_size.py --leakage_type leak_feature --resample_size 200 --resample_seed 0
```
The output of this script is a saved *.npz* file that includes the observed phenotype, the predicted phenotype, and the coefficients. The filename includes additional details, such as the type of leakage, the phenotype, the dataset, the number of cross-validation folds, and the random seed. 

# Sample size analysis

Our sample size analysis is in the file [leakage_sample_size.py](leakage_sample_size.py)

You can input the type of leakage you want to perform, the start/end random seeds, the number of cross-validation folds k, the percentage of features to use, and the model type. Arguments are further detailed in the script, but an example call is below:

```
python run_leakage.py --leakage_type leak_feature --model_type ridge
```
The output of this script is a saved *.npz* file that includes the observed phenotype and the predicted phenotype. The filename includes additional details, such as the type of leakage, the phenotype, the dataset, the number of cross-validation folds, the random seed, and the sample size. 

# Family leakage analysis

Our family analysis scripts are in the folder [family_analysis](family_analysis).

We ran a twin analysis in the file [run_twin_analysis.py](https://github.com/mattrosenblatt7/leakage_neuroimaging/blob/main/family_analysis/run_twin_analysis.py).

You can input the phenotype and the seed.

```
python run_twin_analysis.py --pheno age --seed 0
```
The output of this script is a saved *.csv* file that includes the phenotype, the model type, the type of leakage, the random seed, and the performance (Pearson's r).  

\
We ran a simulation analysis varying the percentage of participants belonging to a family with multiple members in the dataset in the file [run_family_leakage_simulation.py](https://github.com/mattrosenblatt7/leakage_neuroimaging/blob/main/family_analysis/run_family_leakage_simulation.py).

You can input the phenotype and the seed.

```
python run_family_leakage_simulation.py --pheno age --seed 0
```
The output of this script is a saved *.npz* file that includes the observed phenotype and the predicted phenotype. The filename includes additional details, such as the percentage of participants coming from a multi-member family, whether there is family leakage, the phenotype, the dataset, the number of cross-validation folds, and the random seed. 


# Plotting

To make plots, our summary data are included in the [data_for_plots](data_for_plots) folder. You can view the notebook [make_plots.ipynb](make_plots.ipynb), which also has a Google Colab link:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mattrosenblatt7/leakage_neuroimaging/blob/main/make_plots.ipynb)

# Data

Summary data from our experiments are included in the [data_for_plots](data_for_plots) folder. 

* Adolescent Brain Cognitive Development Study (NIMH Data Archive, https://nda.nih.gov/abcd)
* Healthy Brain Network (International Neuroimaging Data-sharing Initiative, https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/)
* Human Connectome Project (ConnectomeDB database, https://db.humanconnectome.org)
* Philadelphia Neurodevelopmental Cohort (dbGaP Study, accession code: phs000607.v3.p2, https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000607.v3.p2)

Data collection was approved by the relevant ethics review board for each of the four datasets.

# License

This code is covered under the MIT License.
