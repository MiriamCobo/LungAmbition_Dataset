# Lung Ambition project, Pamplona-ELCAP dataset

The code to process and run technical validation experiments.

## Data acquisition

Access to the dataset on Zenodo will be granted upon the signing of the institutional agreement https://doi.org/10.5281/zenodo.15120062 

## Enviroment configuration

```
conda env create -f lungAmbpy3.yml
```

## Image Models

Unzip LDCT_data.zip and set up in your computer. We recommend you use at least one GPU. Prepare config.json, process data as in \DataPreparation, creating the 3 folds and run the models.

## Protein models

Download Olink_proteomics.xlsx, process data as in \DataPreparation and run the models.

## Citation

If you use this dataset or code in your research, please cite:
* Dataset:
```
Cobo, Miriam, et al. "Lung Ambition dataset" [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15120062
```
* Paper:
```
Cobo, Miriam, et al. "A multimodal dataset for personalized low-dose CT-based lung cancer screening research". Under revision
```