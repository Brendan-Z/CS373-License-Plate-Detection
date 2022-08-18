# CS373 Assignment

## About

This is a image processing assignment for the course [COMSCI 373](https://courseoutline.auckland.ac.nz/dco/course/COMPSCI/373). The aim of this project is to detect the bounding box around the license plate in an
image of a car. The project also includes an extension script which aims to further extract the characters of the license plate using an OCR algorithm.

Install requirements:

```bash
> pip install -r requirements.txt
```

Run the main program:

```bash
> python CS373LicensePlateDetection.py <image_file>.png
```

Run the extension program:

```bash
> python CS373_extension.py <image_file>.png
```

Run on all images with the run script:

```bash
> python run.py
```

```bash
> python run.py -extension
```
## Technologies

- [matplotlib](https://github.com/matplotlib/matplotlib) for plotting output.
- [numpy](https://github.com/numpy/numpy) for numerical computation.
- [opencv](https://github.com/opencv/opencv-python) for image processing.
- [easy-ocr](https://github.com/JaidedAI/EasyOCR) english OCR algorithm.
