# Face Emotion Recognition using Python and CNN


This repository contains Python code for facial emotion recognition using Convolutional Neural Networks (CNN). The current implementation focuses on detecting two emotions: **happy** and **surprised**, with associated accuracy percentages.

## Overview

Our program leverages a pre-trained model for face recognition and then adds additional layers to create a model capable of detecting emotions. Specifically, we train the model to identify whether a person's emotion is happy or surprised. Surprised is detected when the person covers their mouth with their hands in surprise.

## Getting Started

Follow these steps to run the program:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/jithu-francis/Face-Emotion-Recognition.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Face-Emotion-Recognition
   ```

3. Run the main Python script:

   ```bash
   python main.py
   ```

   This script loads the pre-trained model and you can feed your live video stream and check your emotions.

## Requirements

Make sure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- Keras
- OpenCV (for image processing)
- Numpy
- Matplotlib (for visualization)

You can install these dependencies using `pip`:

```bash
pip install tensorflow keras opencv-python numpy matplotlib
```

## Model Architecture

Our model is based on a Convolutional Neural Network (CNN) architecture, which is widely used for image classification tasks. The details of the architecture can be found in the `Emotion_Detection_Model.ipynb` file.

## Dataset

We used a custom dataset for training and testing our emotion recognition model. The dataset consists of images of individuals expressing happy and surprised emotions, with labels corresponding to these emotions. Please note that this dataset is not publicly available in this repository.

## Model Training

Training the model is not included in this repository due to the absence of the dataset. However, you can train the model on your own dataset by following these general steps:

1. Prepare a dataset with labeled images of happy and surprised emotions.

2. Split the dataset into training and testing sets.

3. Use the provided model architecture in `Emotion_Detection_Model.ipynb` as a starting point and train the model using your dataset.

4. Save the trained model weights.

## Results

Our model detects happy face and sad face with it's accuracy percentage along with it.

## Customization

You can extend this project by:

- Adding more emotions to detect.
- Training the model on a larger and more diverse dataset.
- Improving the model architecture for better performance.
- Building a user interface for real-time emotion recognition from a webcam feed.

Feel free to explore and enhance the capabilities of this facial emotion recognition system!

If you have any questions or suggestions, please feel free to [contact us](mailto:jithufrancis2000@gmail.com).

Thank you for using our facial emotion recognition system!
