# Quantification of bioluminescent pathogens in *Arabidopsis thaliana*
This pipeline provides a tool to batch process images of Arabidopsis rosettes that have been infected with bioluminescent plant pathogens such as vascular *Xanthomonas campestris* pv. campestris and the mesophyll pathogen *Pseudomonas syringae* pv. tomato. The tool can analyze both plant disease symptoms and pathogen colonization through bioluminescence. It was developed at the department of Molecular Plant Pathology at the University of Amsterdam, the Netherlands.

## Contents
This repository contains a number of files that are needed to run the entire analysis:
1. Dependencies to install:
	- `requirements.txt`
2. Environment file with analysis parameters:
	- `example.env`
3. Image input folder
	- `example_input`
4. Scripts to run:
	- `analyze.py`
	- `calibrator.py`
	- `image.py`
	- `settings.py`

### 1. Installation and dependencies 
To get started, download or clone this repository to a location on your personal machine. It is advised to use an environment manager such as [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) to install all dependencies. 

Create a virtual conda environment in which you install your dependencies:

```
conda create -- name environment_name python == 3.11

```
Next, activate the environment:
```
conda activate environment_name

```
Now, install the dependencies within the environment (check if terminal shows *environment_name* in front of the location):
```
conda install -r requirements.txt

```
Whenever you want to leave the environment, type:
```
conda deactivate
```

### 2. Environment file with parameters
This file `example.env` contains all parameters needed to run the analysis on your images. You can adjust the name and then the parameters to your own needs:
- `BLUR`: Amount of blur to be applied in image preprocessing.
- `MORPH`: Kernel size for morphing during preprocessing.
- `ITERATIONS`: Iteration count for image morphing during preprocessing.
- `GREEN/YELLOW`: Ranges specifying color values to be interpreted as green or yellow.
- `CROP_H_MIN/MAX`: Specify horizontal edges of the image's region of interest.
- `THRESHOLD`: Minimum area for a rosette to be recognized.
- `NOISE_THRESHOLD_MIN`: Minimum size of a luminescent object to be considered *no noise*. If it falls below this threshold, it's removed from the analysis.
- `NOISE_THRESHOLD_MAX`: Maximum size of a luminescent object to be considered *no noise*. If it goes above this threshold, it's removed from the analysis.
- `MERGED_THRESHOLD_MAX`: Maximum area for the merged result of two contours (plant rosettes). If rosette contours overlap, they can be considered as one object. You can avoid this by setting this maximum rosette size.
- `GAMMA`: Value to be applied when performing gamma correction when the image is too dark to find individual rosettes.
- `SATURATION`: Factor by which to multiply saturation. Integers only for now, fix coming soon.
- `ROWS`: Number of rows in the tray.
- `COLUMNS`: Number of columns in the tray.
- `INPUT_DIR`: Directory containing images to be analyzed. Find expected file names below. Must have a trailing `/`.
- `CSV_DIR`: Directory to store results. Will be created if necessary. Must have a trailing `/`.
- `CALIBRATION`: The perspective tranformation matrix used to map an ES image onto an RGB image. If this is not given, the calibrator.py script needs to be run on calibration images to calculate the matrix.

### 3. Image input folder
This folder contains all the input images to be analyzed. All images in the folder will be batch-processed when running the analysis script. For each sample, the analysis requires two images that **match**: an RGB (plant) image and a CCD (luminescence) image. These files should be names as follows:
- `XX_DPI_Y_RGB.png` for the RGB image and `XX_DPI_Y_ES.png` for the CCD image
	- `XX` is either `07` or `14`. For hydathode infections, a specific `07` DPI environment file can be created to count individual objects. If you want to analyze total luminescence, use `14` DPI.
	- `Y` is any unique string of digits identifying a pair of images. 
	
### 4. Scripts to run

1. (*optional*) `calibration.py`
	- This file only needs to be run if no calibration matrix to match the RGB and CCD images is in the .env file yet!
	- *Input*: An RGB image and a CCD image of a standard calibration sheet (such as 4 x 11 asymmetric dots)
	- *Run*: `./calibration.py <path_to_rgb_file> <path_to_es_file> <path_to_env>`
	- *Output*: If no entry named `CALIBRATION` exists in the .env file, an entry will be created when running this script. If an entry already exists, it will be overwritten.
	
2. `analyze.py`
	- This is the analysis file and when running this, it automatically incorporates the `image.py` and `settings.py` scripts.
	- *Input*: All requirements mentioned above (dependencies, environments and input images).
	- *Run*: `./analyze.py [path_to_env]`
	- *Output*: an overlay image and .csv file containing all quantified parameters in the `CSV_DIR` specified in the .env file.


## Citation
If this pipeline was useful for your work, please cite:
```
Nanne W. Taks, Mathijs Batstra, Ronald Kortekaas, Floris D. Stevens, Sebastian Pfeilmeier and Harrold A. van den Burg, **Visualization and quantification of spatiotemporal disease progression in Arabidopsis using a bioluminescence-based imaging system** *Journal title* (2024).
```

