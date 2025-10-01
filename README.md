
# Vision Extraction

Vision Extraction is a web-based application that performs object segmentation on images using a DeepLabV3 model. Users can upload an image and obtain a predicted segmentation mask highlighting objects of interest.



## Description

Vision Extraction is a cutting-edge web application designed to perform semantic segmentation on digital images. It empowers users to identify and isolate objects within an image by generating pixel-level classification masks using the powerful DeepLabV3 deep learning model. This tool is especially valuable for researchers, developers, and enthusiasts working in computer vision, AI, and image analysis.

## Features

- Upload images (`jpg`, `jpeg`, `png`) for object extraction.
- Side-by-side display of original vs predicted images.
- Download the extracted objects as PNG.
- Works on CPU or GPU.
- Built with an interactive Streamlit interface.


## Project Structure

VISION EXTRACTION/

│

├── app.py

├── requirements.txt

├── README.md

├── models/

│   └── best_model11.pth

└── .gitignore
## Installation

1. Clone the project or copy files locally.
2. Install dependencies:

```bash
pip install -r requirements.txt
````

3. Ensure `best_model11.pth` and `Explainer.png` are in the project folder.

## Usage

Run the Streamlit app:

```bash
streamlit run VisionExtraction.py
```

* Upload an image via the interface.
* View original and predicted segmentation side-by-side.
* Download the predicted image.
## Notes


* The model currently predicts two classes: background and object.
* Make sure the model and image files are in the correct locations.
* Adjust preprocessing or model paths if using a different model.

## Dependencies


* Python 3.x
* Streamlit
* PyTorch
* Torchvision
* NumPy
* Pillow


## License

MIT License

Copyright (c) 2025 springboardmentor468a

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.