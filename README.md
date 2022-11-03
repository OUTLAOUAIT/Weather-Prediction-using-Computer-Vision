# Weather-Prediction-using-Computer-Vision

## Objective

In this project, I will try to estimate the atmospheric temperature at Rainbow Bridge, Tokyo by looking at the images of this location.

## Method

I will be using a convolutional neural network to train a model that predicts the temperature at the moment the image was taken. Our task will be a regression task as our target variable (temperature) is a continuous variable.

## Data collection

For this task, we will be connecting to Youtube and OpenWeatherMap APIs for collecting images and weather information, respectively. We write a script that connects to Youtube API and takes a screenshot of the Odaiba Live Camera feed once every 50 seconds. Once the snapshot is taken, the script connects to OpenWeatherMap API and gets the weather information at Rainbow Bridge (Odaiba) The collected data is written to our local machine together with a timestamp to link back the image and temperature later. We run the script for roughly 2 days and collect 2k+ images together with the weather information.

## Data processing
When OpenWeatherMap API is requested to provide weather information about a coordinate, it checks the nearby weather stations and provides the latest info. For example, when we query about Rainbow Bridge, we get information from Mita or Shinagawa station depending on the time the query was done. Since we do not want discontinuity in our data (Mita and Shinagawa might have different temperatures at the same time) we narrow down the dataset to information obtained from the Mita station. This leaves us with 1,902 images in total. Image file paths and relevant temperature together with the timestamp are collated into a master data file. This data together with images are uploaded to Google Drive for training the model on Google Colab.

## Model
Using 1,902 images of the Rainbow Bridge area, we divide the dataset into training (70%) and test (30%) subsets. A portion of the training subset will be used as a validation subset for early stopping in order to prevent overfitting.

The neural network is a hybrid deep learning made up of a VGG16, 2 convolutional+MaxPool layers + 1 Global Average Pooling layer + 3 dense layers, and 1 output layer (temperature).

The convolutional layer slides a small window across the image and extracts useful features from the image. In a commonly used example, a convolutional layer can learn to identify edges in an image. Applying MaxPool right after the convolutional layer is common practice for downsizing the output. We then apply the Global Average Pooling layer to reduce the image to a 1D vector. This 1D vector is moved through a standard 3-layered multilayer perceptron with a ReLu activation function that will give us the temperature as 1 value from the image.

We use GPU acceleration on Google Colab for faster learning. The first epoch will take a relatively long time as tensors are moved to GPU, but future epochs will take a significantly shorter time and learning will be finished in less than 10 mins.

## Results
The model reaches a Root Mean Squared Error of  0.23 °C, while the R-squared is 0.9911
