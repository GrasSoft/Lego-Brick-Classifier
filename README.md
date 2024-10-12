# LEGO BRICKS CLASSIFIER PROJECT
Hi and welcome to the github repository of the LEGO Bricks Classifier project. 
In this README file, the codes and repositories used for this project are shortly denoted.

### Owners and Date
- Cristian Cutitei @GrasSoft
- Bakul Jangley @bakuljangley
- Tjerk van der Weij @tvanderweij

*15/06/2024*

## Datasets used

The original LEGO Dataset (single bricks): https://www.kaggle.com/datasets/ronanpickell/b200c-lego-classification-dataset
The original LEGO Dataset (piles of bricks): https://www.kaggle.com/datasets/ronanpickell/b100-lego-detection-dataset

The modified dataset (with occlusion and brightness): https://www.icloud.com/iclouddrive/05erFwWmkl-e3g9lzlfRlejVg#modifiedDataset

## Scripts

### These python scripts are used to modify the dataset.

`changeBrightness.py` - this script will create a separate folder containing images from the dataset with different level o brightness changed, it will either increse or decrease the brightness levels by 20% or 50%.

`occlude.py` - this script will create a separate folder containing images from the dataset with a 12 by 12 black "occlusion" box that will randomly cover a part of the image of the lego brick.

`createModifiedTrainingset.py` - this script will create a separate folder containing images from the dataset, it will randomly choose between the aforementioned functions (occlude and changeBrightness) and add it along with the original image to a separate folder.

`removenoise.py` - this script will remove noise from an input image by applying (inverse) fourier transforms.

### The following scripts are for the bounding boxes:

`FourierAnalysis.py` - this script creates a .tiff float32 image with normalized values (between 0 and 1) with the magnitude spectrum (for later editting in GIMP) and a .npy file with the phases. Also plots the 2 files, for visualization.

`InverseFourier.py` - this scripts is to visualize the inverse fourier trasform of the .tiff(magnitude) and .npy (phases) from the previous script.

`adaptive_thresholding.py` - this script will apply adaptive thresholding to an input image, the output can be seen in adaptive_thresholding.png.

`blob_detection.py` - this script will apply blob detection to an input image, the output can be seen in blob_detection.png.

`canny_edge_detection.py` - this script will apply basic canny edge detection to an input image, the output can be seen in canny_edge_detection.png.

`edge_detection.py` - this script will apply advanced edge detection with morphological operations to an input image and draw bounding boxes, the output can be seen in bounding_boxes_with_canny_edge_detection.png.

### The following scripts are for the results:

`Experiments.ipynb` - running this jupyter notebook file will give model results when training on the original dataset like confusion matrices, accuracy, precision and f1 score.

`ModifiedDatasetExperiments.ipynb` - running this jupyter notebook file gives results when training on the modified dataset model results like confusion matrices, accuracy, precision and f1 score.

`createplot.py` - running this script will create the desired plots.

`resnet34_noocclusion_bigdataset.zip` and `resnet34_occlusion_bigdataset.zip` - the zip files containing the results of resnet34 on the big dataset.

## Models
`models.py` - in this script the torch classes of custom resnet18, resnet34 and resnet50 can be seen.

## Repositories

`cutouts` repository - cutouts of image of piles of bricks can be found here, made by using the edge_detection.py file.

`model_results` repository - preliminary results can be found here.

`resnet18` repository - final big dataset results of resnet18_brightness, resnet18_normal and resnet18_occlusion can be found here.

`resnet34_colour`, `resnet34_noocclusion` and `resnet34_occluded` repositories - results for the resnet34 models can be found here.

`resnet50` repository - final big dataset results of resnet50_brightness, resnet50_normal and resnet50_occlusiion can be found here.





