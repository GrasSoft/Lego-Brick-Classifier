# LEGO BRICKS CLASSIFIER PROJECT
Hi and welcome to the github repository of the LEGO Bricks Classifier project. 
In this README file, the codes and repositories used for this project are shortly denoted.

### Owners 
- Cristian Cutitei @GrasSoft


## Datasets used

The original LEGO Dataset (single bricks): https://www.kaggle.com/datasets/ronanpickell/b200c-lego-classification-dataset
The original LEGO Dataset (piles of bricks): https://www.kaggle.com/datasets/ronanpickell/b100-lego-detection-dataset

## Scripts

### These python scripts are used to modify the dataset.

`createModifiedTrainingset.py` - this script will create a separate folder containing images from the dataset, it will randomly choose between the aforementioned functions (occlude and changeBrightness) and add it along with the original image to a separate folder.

### The following scripts are for the results:

`Experiments.ipynb` - running this jupyter notebook file will give model results when training on the original dataset like confusion matrices, accuracy, precision and f1 score.

`ModifiedDatasetExperiments.ipynb` - running this jupyter notebook file gives results when training on the modified dataset model results like confusion matrices, accuracy, precision and f1 score.

`resnet34_noocclusion_bigdataset.zip` and `resnet34_occlusion_bigdataset.zip` - the zip files containing the results of resnet34 on the big dataset.

## Models
`models.py` - in this script the torch classes of custom resnet18, resnet34 and resnet50 can be seen.
