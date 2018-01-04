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

[image4]: ./new_images/stop.jpeg "Stop"

[image5]: ./new_images/road_work.jpeg "Road Work"

[image6]: ./new_images/keep_right.jpeg "Keep Right"

[image7]: ./new_images/70.jpeg "70"

[image8]: ./new_images/children_crossing.jpeg "Children Crossing"

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

* The first architecture that was chosen is LeNet. It is a proven LeNet is a famous CNN structure for image recognition.

* The problem is LeNet is that it has only 2 Convolutional Layers and the size of its layers are too small which has onnly 6 and 16 filters respectively.

* I increase the number of Layers to 4, each of which has 128 filters. In addition, for each Convolution Layer, I also added a average pooling layer, a dropout layer.

* I changed the default batch size from 128 to 64 and also tried various dropout rate from 0.5, 0.75 and the final value of 0.85. These changes resuled in better Validation and Test accuracies. I also dramatically reduced the number of training Epochs from 500 to 30, because I found that the validation accuracy barely changed has saturated after Epoch 30.

* The introdution of the dropout layer increase the test accuracy as well as the model's capability of predicting new images. The dropout layer is a kind of regulation to the model to prevent overfitting. I forces each neuron in the layer to be more independently capable of extracting features.
 

### Test the Model on new images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...


### Predictions on new images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Road Work     		| Road Work 									|
| Keep Right			| Keep Right									|
| 70 km/h	      		| General caution					 			|
| Children Crossing		| Children Crossing	     						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is worse than the accuracy on the test set of 96.9%. The reason is that for the the image of 70 km/h, the object was too blur after preprocessing and it is too far from the center of the image.


#### Prediction on new images

For the first image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .0463         		| Stop   										| 
| .0277  				| Priority road 								|
| .0268					| Speed limit (120km/h)							|
| .0262	      			| Turn left ahead					 			|
| .0260				    | Speed limit (70km/h)     						|

For the second image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .0442         		| Road Work   									| 
| .0304  				| Road narrows on the right						|
| .0303					| Bicycles crossing								|
| .0298	      			| General caution					 			|
| .0282				    | Bumpy road    								|

For the third image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .0481         		| Keep right  									| 
| .0292  				| End of speed limit (80km/h)					|
| .0280					| General caution								|
| .0278	      			| Roundabout mandatory					 		|
| .0272				    | Priority road   								|


For the fourth image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .0405         		| General caution  								| 
| .0371  				| Traffic signals								|
| .0350					| Priority road									|
| .0324	      			| Road work					 					|
| .0318				    | Road narrows on the right    					|


For the fifth image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .0470         		| Children crossing  							| 
| .0372  				| Bicycles crossing								|
| .0333					| Slippery road									|
| .0324	      			| Beware of ice/snow					 		|
| .0278				    | General caution    							|