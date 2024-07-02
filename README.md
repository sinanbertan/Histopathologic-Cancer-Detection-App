
# Histopathologic Cancer Detection

This project aims to detect cancer in histopathological images using deep learning techniques. Cancer is a disease where early diagnosis and correct treatment methods are critical. Histopathological images are images obtained by examining tissues under a microscope and play an important role in cancer diagnosis. While analysis performed with traditional methods can be time-consuming and tiring, deep learning techniques can speed up this process and increase its accuracy.
In the project, a model was developed using deep learning algorithms such as convolutional neural networks (CNN). This model is trained to detect cancerous cells in histopathological images. The generalization ability of the model was increased by using various data augmentation techniques for training the model. A reliable and comprehensive histopathological image dataset was used as the data set.
The performance of the model was evaluated with accurcay and loss function. The results obtained show that the model can detect cancerous cells with high accuracy. This proves the usability and effectiveness of deep learning techniques in histopathological analysis.
A website was designed to show the outputs of the model created in the project. Through the designed website, users will be able to enter a tissue sample and determine whether there is cancer in the entered tissue. The application is built with Flask, and the frontend is developed using HTML, CSS, and JavaScript.
## Acknowledgements

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)
## Project Overview

The Cancer Detection App aims to provide a tool for early detection of cancer by analyzing histopathological images. The CNN model has been trained on a dataset of microscopic images and can classify images as cancerous or non-cancerous with high accuracy.
## Data Source
You can access the data source I have used from here:
https://www.kaggle.com/c/histopathologic-cancer-detection/data
## Features

- Upload histopathological images for analysis
- Get predictions on whether the image is cancerous or non-cancerous
- View prediction results on the web interface
- User-friendly UI for easy interaction

## Technologies Used

- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Deep Learning:** TensorFlow, Keras, scikit-learn, SciKeras
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Plotly, Seaborn
## Installation

To run this project locally, follow these steps:

```sh
git clone https://github.com/sinanbertan/Histopathologic-Cancer-Detection-App.git
cd Cancer_Detection
```
* Create and Activate a Virtual Environment:

```sh
python -m venv venv
source venv/bin/activate
```
* Install the Required Packages:
```sh
pip install -r requirements.txt
````

* Set Up the Flask Environment:
```sh
export FLASK_APP=app.py
export FLASK_ENV=development
````
* Run the Application:
```sh
python -m flask run
````
* Open Your Browser:
Navigate to http://127.0.0.1:5000 in your web browser to view and interact with the application
## Usage/Examples

* Open the web application in your browser.
* Upload a histopathological image using the upload button.
* Click on the "Analyze" button to get the prediction results.
* View the results displayed on the web page.



## Model Training

The CNN model was trained using the following steps:

*  Preprocessing the histopathological images using CNN.
* Splitting the dataset into training and validation sets.
* Defining the CNN architecture using Keras and TensorFlow.
* Training the model on the training set and validating it on the validation set.
* Saving the trained model for inference in the web application.
## Screenshots

#### Block theme of CNN model:
![App Screenshot](https://github.com/sinanbertan/Histopathologic-Cancer-Detection-App/blob/main/Cancer_Detection/static/assets/Cnn.drawio.png)

#### Home page of website:
![App Screenshot](https://github.com/sinanbertan/Histopathologic-Cancer-Detection-App/blob/main/Cancer_Detection/static/assets/home.png)

#### Uploading an image to the website:
![App Screenshot](https://github.com/sinanbertan/Histopathologic-Cancer-Detection-App/blob/main/Cancer_Detection/static/assets/home1.png)

#### Result page: 
![App Screenshot](https://github.com/sinanbertan/Histopathologic-Cancer-Detection-App/blob/main/Cancer_Detection/static/assets/result.png)

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please open an issue or submit a pull request.
## License

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License - see the LICENSE file for details.


