# Melanoma Skin Cancer Detection

## Abstract

In the realm of cancer, there exist over 200 distinct forms, with melanoma standing out as the most lethal type of skin cancer among them. The diagnostic protocol for melanoma typically initiates with clinical screening, followed by dermoscopic analysis and histopathological examination. Early detection of melanoma skin cancer is pivotal, as it significantly enhances the chances of successful treatment. The initial step in diagnosing melanoma skin cancer involves visually inspecting the affected area of the skin. Dermatologists capture dermatoscopic images of the skin lesions using high-speed cameras, which yield diagnostic accuracies ranging from 65% to 80% for melanoma without supplementary technical assistance. Through further visual assessment by oncologists and dermatoscopic image analysis, the overall predictive accuracy of melanoma diagnosis can be elevated to 75% to 84%. The objective of the project is to construct an automated classification system leveraging image processing techniques to classify skin cancer based on images of skin lesions.

## Problem statement

To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

## Table of Contents
* [General Info](##general-information)
* [Model Architecture](##model-architecture)
* [Technologies Used](##technologies-used)
* [Conclusions](##conclusions)
* [Acknowledgements](##acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information

The dataset comprises 2357 images depicting malignant and benign oncological conditions, sourced from the International Skin Imaging Collaboration (ISIC). These images were categorized based on the classification provided by ISIC, with each subset containing an equal number of images.

In order to address the challenge of class imbalance, the Augmentor Python package (https://augmentor.readthedocs.io/en/master/) was employed to augment the dataset. This involved generating additional samples for all classes, ensuring that none of the classes had insufficient representation.

The aim of this task is to assign a specific class label to a particular type of skin cancer.

## Model Architecture

The break down of the final provided CNN architecture step by step:

1. **Normalization**: The `Rescaling(1./255)` layer is added to normalize the pixel values of the input images. Normalization typically involves scaling the pixel values to a range between 0 and 1, which helps in stabilizing the training process and speeding up convergence.

2. **Convolutional Layers**: Three convolutional layers are added sequentially using the `Conv2D` function. Each convolutional layer is followed by a rectified linear unit (ReLU) activation function, which introduces non-linearity into the model. The `padding='same'` argument ensures that the spatial dimensions of the feature maps remain the same after convolution. The number within each `Conv2D` layer (32, 64, 128) represents the number of filters or kernels used in each layer, determining the depth of the feature maps.

3. **Pooling Layers**: After each convolutional layer, a max-pooling layer (`MaxPooling2D`) is added to downsample the feature maps, reducing their spatial dimensions while retaining the most important information. Max-pooling helps in reducing computational complexity and controlling overfitting.

4. **Dropout Layer**: A dropout layer (`Dropout`) with a dropout rate of 0.5 is added after the last max-pooling layer. Dropout is a regularization technique used to prevent overfitting by randomly dropping a fraction of the neurons during training.

5. **Flatten Layer**: The `Flatten` layer is added to flatten the 2D feature maps into a 1D vector, preparing the data for input into the fully connected layers.

6. **Fully Connected Layers**: Two fully connected (dense) layers (`Dense`) are added with ReLU activation functions. The first dense layer consists of 128 neurons, and the second dense layer outputs the final classification probabilities for each class label.

7. **Output Layer**: The number of neurons in the output layer is determined by the `target_labels` variable, representing the number of classes in the classification task. The output layer does not have an activation function specified, as it is followed by the loss function during training.

8. **Model Compilation**: The model is compiled using the Adam optimizer (`optimizer='adam'`) and the categorical_crossentropy as loss function, which is suitable for multi-class classification problems. Additionally, accuracy is chosen as the evaluation metric (`metrics=['accuracy']`).

9. **Training**: The model is trained using the `fit` method with the specified number of epochs (`epochs=20 and 30`).

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
- Conclusion 1 from the analysis: In the Initial model- The Training accuracy shows steady increase however ,its poor performance on validation data indicates a lack of generalizability.
- Conclusion 2 from the analysis: By leveraging augmented data,overfitting addressed to a very little extent.Both the training and validation data are showing poor performance, indicating that the model is now exhibiting underfitting.
- Conclusion 3 from the analysis: Applied Class imbalance and trained the model , observed that Very high training accuracy suggests that the model is fitting the training data almost perfectly. Hence it is indicating the model is clearly Overfitting.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- Matplotlib - version 3.10.0
- Numpy - version 1.26.4
- Pandas - version 2.2.2
- TensorFlow - version 2.18.0

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements
- UpGrad tutorials on Convolution Neural Networks (CNNs) on the learning platform

## Collaborators
- Created by [Ramu Pallepati](https://github.com/ramuautomation)
