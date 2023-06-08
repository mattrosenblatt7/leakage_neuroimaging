This repository includes data and code for the preprint [*The effects of data leakage on neuroimaging predictive models*](https://github.com/mattrosenblatt7/leakage_neuroimaging)

The following code and data are for non-commercial and academic purposes only.

# Main leakage analysis

The main leakage analysis is found in the file [run_leakage.py](run_leakage.py)

You can input the type of leakage you want to perform, the number of cross-validation folds k, the percentage of features to use, the sample size you want, and the seed for resampling. Arguments are further detailed in the script, but an example call is below:

```
python leakage_sample_size.py --leakage_type leak_feature --resample_size 200 --resample_seed 0
```

# Sample size analysis

Our sample size analysis is in the file [leakage_sample_size.py](leakage_sample_size.py)

You can input the type of leakage you want to perform, the start/end random seeds, the number of cross-validation folds k, the percentage of features to use, and the model type. Arguments are further detailed in the script, but an example call is below:

```
python run_leakage.py --leakage_type leak_feature --model_type ridge
```

# Plotting

To make plots, our summary data are included in the [data_for_plots](data_for_plots) folder. You can view the notebook [make_plots.ipynb](make_plots.ipynb), which also has a Google Colab link:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mattrosenblatt7/leakage_neuroimaging/blob/main/make_plots.ipynb)
