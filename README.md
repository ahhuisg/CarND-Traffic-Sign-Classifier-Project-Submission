# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./doc/dataset_stat.png "Data Set Stat"

[image2]: ./doc/before_gray.png "Before Grayscaling"

[image3]: ./doc/after_gray.png "After Grayscaling"

[image4]: ./doc/augt_org.png "Original Image"

[image5]: ./doc/augt_1.png "Augmented Image 1"

[image6]: ./doc/augt_2.png "Augmented Image 2"

### Data Set Summary & Exploration

#### Basic summary of the data set

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a chart showing how the data is distributed among the 43 classes

![alt text][image1]

### Design and Test a Model Architecture

1. At 1st step I decided not to do any pre-processong of the training images and see what is the result. The valuation and the testing accuracy is very poor, which is **0.054** and **0.059** respectively. Here is the [link](./notebooks/Traffic_Sign_Classifier-No-Preprocessing.ipynb) to the notebook

2. At 2nd step, I decided to just do a **Shuffling** of the test image. The valuation and testing accuracy is surprisingly good, which **0.943** and **0.928** respectively. Here is the [link](./notebooks/Traffic_Sign_Classifier-Shuffle.ipynb) to the notebook

3. At the 3rd step, besides **Shuffling**, I decided to test out different hyperparameters. At each test, 1 hyperparameter is changed and others remain unchanged. I change the **epochs to 1000** and **batch size to 256** as well as **learning rate to 0.002 and 0.01 respectively**. It turns out that the new hyperparameters actually make the valuation and testing accuracy worse. Here are the links to the notebooks: [epechs 1000](./notebooks/Traffic_Sign_Classifier-Shuffle-1000epochs.ipynb), [batch size 256](./notebooks/Traffic_Sign_Classifier-Shuffle-batch256.ipynb), [learning rate 0.01](./notebooks/Traffic_Sign_Classifier-Shuffle-lr-0.01.ipynb) and [learning rate 0.002](./notebooks/Traffic_Sign_Classifier-Shuffle-lr-0.002.ipynb)

4. At the 4th step, besides **Shuffling**, I decided to convert the images to **Grayscale** because the coloring of the text does not affect the visual recognition of the shape of the targeted object. The images below are an example of a traffic sign image before and after grayscaling. The valuation and testing accuracy is **0.946** and **0.922** respectively. Here is the [link](./notebooks/Traffic_Sign_Classifier-Shuffle-Gray.ipynb) to the notebook

![alt text][image2] ![alt text][image3]

5. At the 5th step, besides **Shuffling** and **Graying**, I decided to the **Normalize** the input image as well. The valuation and testing accuracy is **0.942** and **0.923** respectively. Here is the [link](./notebooks/Traffic_Sign_Classifier-Shuffle-Normalize-Gray.ipynb) to the notebook

6. For my final architecture and testing, besides **Shuffling**, **Graying** and **Normalize**, I decided to generate additional data. For each Training input image, I generated **40 more images**, 20 of which are **ramdonly rotated image between -60 and 60 degrees** respectively. The other 20 are **shifted images of between -6 for X and 6 for Y pixels** respectively. The valuation and testing accuracy is **0.974** and **0.969** respectively. Here is the [link](./Traffic_Sign_Classifier.ipynb) to the notebook


#### Final Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x128				|
| Dropout	      		| 0.85											|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x128					|
| Dropout	      		| 0.85											|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 8x8x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x256					|
| Dropout	      		| 0.85											|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 4x4x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x128					|
| Dropout	      		| 0.85											|
| Fully Connected 		| Input = 512, Output = 1024 					|
| RELU					|												|
| Fully Connected 		| Input = 1024, Output = 1024 					|
| RELU					|												|
| Fully Connected 		| Input = 1024, Output = 1024 					|
| RELU					|												|
| Fully Connected 		| Input = 1024, Output = 43 					|
 


### Model Hyperparameters

**Optimizer: AdamOptimizer**

**Learning Rate: 0.001**

**Batch Size: 64**

**Number of Epochs: 30**


### Training Approach

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.974
* test set accuracy of 0.969


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


