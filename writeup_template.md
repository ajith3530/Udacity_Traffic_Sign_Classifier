# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Writeup_Images/image1.png "Visualization"
[image2]: ./examples/Writeup_Images/image2.png "Grayscaling"

[image4]: ./examples/Writeup_Images/image4.png "Traffic Sign 1"
[image5]: ./examples/Writeup_Images/image5.png "Traffic Sign 2"
[image6]: ./examples/Writeup_Images/image6.png "Traffic Sign 3"
[image7]: ./examples/Writeup_Images/image7.png "Traffic Sign 4"
[image8]: ./examples/Writeup_Images/image8.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

| Tables   |      Are      |  Cool |
|----------|:-------------:|------:|
| col 1 is |  left-aligned | $1600 |
| col 2 is |    centered   |   $12 |
| col 3 is | right-aligned |    $1 |

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

First the images were converted to dtype float to ensure the value of the individual pixels are normalized between 0 and 1. This helps  reduce computational time and power. 
Then the images were grayscaled, to prevent high contrast pixel regions affecting the model's prediction ability.


Here is an example of a traffic sign image before and after normalizing,grayscaling and reshaping.

![alt text][image2]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer    |                     Description                    |
|----------|:--------------------------------------------------:|
| Layer 1  | Convolution Input =  32x32x3 Output = 28x28x6      |
|          |                  Activation - ReLU                 |
|          |   Max Pooling Input = 28x28x6 Output = 14x14x6     |
|          |                                                    |
| Layer 2  | Convolution Input  =  14x14x6 Output = 10x10x16    |
|          |                  Activation - ReLU                 |
|          |   Max Pooling Input = 10x10x16 Output = 5x5x6      | 
|          |            Flatten Input = 5x5x16 Output = 400     |
|          |                                                    |
| Layer 3  | Convolution Input  =  400 Output = 120             |
|          |                  Activation - ReLU                 |
|          |                                                    |
| Layer 4  | Fully Connected Input  = 120 Output = 84           |
|          |                 Activation - ReLU                  |
|          |                                                    |
| Layer 5  | Fully Connected Input  = 84 Output = 43            |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used 30 epochs, a batch size of 128 and a learning rate of 0.0015.

For my training optimizers I used softmax_cross_entropy_with_logits to get a tensor representing the mean loss value to which I applied tf.reduce_mean to compute the mean of elements across dimensions of the result. Finally I applied minimize to the AdamOptimizer of the previous result.

My final model Validation Accuracy was 0.935

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 94.5
* validation set accuracy of 92.3
* test set accuracy of 60.00

If a well known architecture was chosen:
* What architecture was chosen? 
   The LeNet architecture as provided in the Course was selected due to its proven Track Record.
* Why did you believe it would be relevant to the traffic sign application?
   This was a classic case of Transfer Learning wherein the data is kind of similar to Character Recognition and the dataset provided was      very smalled compared to the MNIST dataset.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
   The Training and Validation accuracy momemtum was lost on the Test set, since the model failed to recognize the most of the signs.
   The first root cause which I determined was at the data preprocessing stage, wherein the input image should have been converted to          grayscale and given as an input to the models, as in the grayscale the dynamic parameter(colors) variations are reduced, which in turn      increases the training and validation accuracy as iluustrated in similar projects.  
   The second root cause was the model itself, wherein additional convolutional, dropout layers could have been added to the existing LeNET    architecture to augment its prediction accuracy.
   The third and the most fatal root cause was the lack of availablity of time to test out the theories mentioned in root causes 1 and 2      listed above.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

| Image			        |     Comments	                                                                                    | 
|:---------------------:|:---------------------------------------------:                                                    | 
| Speed Limit 60   		| Since digits 60 is easily distinguishable, it should be easy to predict							| 
| No Entry     			| Since the telltale features can be similar to other signs after resizing, this could be difficult.|
| Speed Limit 50		| Since digits 50 is easily distinguishable, it should be easy to predict                           |
| Speed Limit 120 		| Since digits 120 is similar to 20, it should be difficult to predict                              |
| Stop        			| Since the telltale features can be similar to other signs after resizing, this could be difficult.|

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 60   		| Speed limit (30km/h)							| 
| No Entry     			| Turn right ahead 								|
| Speed Limit 50		| Priority road									|
| Speed Limit 120 		| Speed limit (120km/h)			 				|
| Stop        			| Keep right         							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.
As expected for the 3 difficult images, the model was able to distinguish 1 amongst the 3. This behavior can be improved by augmenting the dataset, by adding external noise and alteration to the dataset to increase the training samples.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Since the model has a pretty bad test accuracy its basically overfitting due to the excess availabilty of dynamic parameters in the image input. This can be addressed by working on resolving Root Cause 1 as list above.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.00%          		| Speed limit (30km/h)							| 
| 100.00%      			| No Entry 								|
| 100.00%        		| Priority road									|
| 000.00%        		| Speed limit (120km/h)			 				|
| 000.00%      			| Stop               							|






