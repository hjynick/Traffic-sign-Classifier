## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)



### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.


The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.


### Basic summary of the data set.
The code for this step is contained in the top code cell  of the IPython notebook.

1. I used the pandas library to calculate summary statistics of the traffic signs data set:

The size of training set is 34799
The size of test set is 12630
The shape of a traffic sign image is (32, 32, 3)
The number of unique classes/labels in the data set is 43

### Exploratory visualization of the dataset.
The code for this step is contained in the code cell #3 of the IPython notebook.
Here is an exploratory visualization of the data set. It is a table structured images with a 4 parameters:
Sign's class name
* Number of occurrences in Training set
* Number of occurrences in Validation set
* Number of occurrences in Test set
![training set][../CarND-Traffic-Sign-Classifier-Project/Code pic/vis1.png]
![bar][../CarND-Traffic-Sign-Classifier-Project/Code pic/vis2bar.png]

### Design and Test a Model Architecture
1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? 
Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

* As a first step, I decided to convert the images to grayscale because in case with traffic signs, the color is unlikely to give any performance boost. But for natural images, color gives extra information, and transforming images to grayscale may hurt performance.
* As a second step, I normalize the image data to reduce the number of shades to increase the performance of the model.
Here is an example of a traffic sign image before and after each step of preprocessing.

My architecture is a deep convolutional neural network inspired by two existing architectures: one is LeNet[1], and the other is the one in Ciresan's paper[3].
Its number and types of layers come from LeNet, but the relatively huge number of filters in convolutional layers came for Ciresan. Another important property of Ciresan's network is that it is multi-column, but my network contains only a single column. It makes it a little less accurate, but the training and predition is much faster.
de cell **#6** of the ipython notebook. 
My final model consisted of the following layers:

|      Layer      |                  Description                  |
| :-------------: | :-------------------------------------------: |
|      Input      |            32x32x1 Grayscale image            |
| Convolution 5x5 | 1x1 stride, 'VALID' padding, outputs 28x28x6  |
|      RELU       |                                               |
|   Max pooling   |         2x2 stride,  outputs 14x14x6          |
| Convolution 5x5 | 1x1 stride, 'VALID' padding, outputs 10x10x16 |
|      RELU       |                                               |
|   dropout       |                                               |
|     Flatten     |                  outputs 400                  |
| Fully connected |                  outputs 120                  |
|      RELU       |                                               |
|                 |                                               |
| Fully connected |                  outputs 84                   |
|      RELU       |                                               |
|                 |                                               |
| Fully connected |               outputs 43 logits               |
|                 |                                               |



```
Layer 1: Convolutional (5x5x1x32) Input = 32x32x1. Output = 28x28x32.
Relu activation.
Pooling. Input = 28x28x32. Output = 14x14x32.

Layer 2: Convolutional (5x5x32x64) Input = 14x14x32. Output = 10x10x64.
Relu activation.
Pooling. Input = 10x10x64. Output = 5x5x64.
Flatten. Input = 5x5x64. Output = 1600.


Layer 3: Fully Connected. Input = 1875. Output = 120.
Relu activation.


Layer 4: Fully Connected. Input = 120. Output = 84.
Relu activation.


Layer 5: Fully Connected. Input = 84. Output = 43.
```

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eleventh cell of the ipython notebook. 

To train the model, I used AdamOptimizer, a batch size of 128, at most 50 epochs, a learn rate of 0.001.
Another hyperparameter was the dropout rate which was 0.7 at every place where I used it. 
I have tried changing these parameters but it didn't really increase the accuracy.
I saved the model which had the best validation accuracy.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the twelfth cell of the Ipython notebook.

My final model results were:

- validation set accuracy of 0.944 
- test set accuracy of 0.936

I started out by creating an architecture which could clearly overfit the training data. (It converged to 1.0 training-accuracy in a couple of epochs, but the validation accuracy was much lower.
Then I have added regulators until the overfitting was more-or-less eliminated. I added dropout operations between the fully connected layers. I also tried L2 regularization for the weights (in addition to the dropout), but it made the accuracy worse by a tiny amount.
Then I have kept removing filters up to the point when the accuracy started decreasing.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I have aquired images from the web.
I cut out the images similarly to the dataset, but I didn't pay attention to the exact size and position of the signs. I programmatically resized them to 32x32 using the function cv2.resize()(not keeping aspect ratio).


###### Photos from the web:

![1][../CarND-Traffic-Sign-Classifier-Project/new traffic/1.jpg]

![2][../CarND-Traffic-Sign-Classifier-Project/new traffic/2.jpg]

This picture is interesting because the perspective and rotation makes the car figures almost form a diagonal similar to the one in End of all speed limits sign.

![3][../CarND-Traffic-Sign-Classifier-Project/new traffic/3.jpg]

![4][../CarND-Traffic-Sign-Classifier-Project/new traffic/4.jpg]

This picture might be hard to classify because it is from a strange perspective and another sign is hanging in to the picture.

![5][../CarND-Traffic-Sign-Classifier-Project/new traffic/5.jpg]

![6][../CarND-Traffic-Sign-Classifier-Project/new traffic/6.jpg]
This is a very blurry image, at first even I didn't know which sign is it. It might be a photo or an artistic painting, of most likely the "Dangerous turn to the left" sign.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the thirteenth cell of the Ipython notebook.

Here are the results of the prediction:

|                Image                    |             Prediction                   |
| :--------------------------------------:| :---------------------------------------:|
| Vehicle over 3.5 metric tons prohibited |  Vehicle over 3.5 metric tons prohibited |
|            keep right                   |          keep right                      |
|            Priority road                |            Priority road                 |
|               stop                      |               stop                       |
|            No Vehicles                  |           No Vehicles                    |
|            Priority road                |            Priority road                 |
|Right of the way in the next intersection| Right of the way in the next intersection|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the thirteenth cell of the Ipython notebook.

For the first image, the model is very confident (97%), and the rest are even higher:

![Top 5 Softmax Probabilities][../CarND-Traffic-Sign-Classifier-Project/Code pic/vis5.png]

```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```
