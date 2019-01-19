# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: writeup_images/nvidia.png "Nvidia Model"
[image2]: writeup_images/center.png "Center"
[image3]: writeup_images/right_to_left.png "Right to left image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* utils.py containing the utility function such as `batch_generator`, `augument`
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. My workflow is simple
```
Get data from dataloader -> Get model architecture -> train_model
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 2x2 filter sizes and depths between 24 and 64 (model.py lines 20-35) 

The model includes combination of RELU and ELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (code line 22). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. Apart from that my model havily uses L2 regulrization with penalty `1e-3` . I am also using Early Stopping concept.

The model was trained and validated on different data sets to ensure that the model was not overfitting. I splitted my model using `sklearn's train_test_split` with `20%` in validation set. I am also using augmentations such as `random_flip`, `random_translate` and `random_brighness`. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, with Keras callback `ReduceOnPlateau` and `EarlyStopping` to reduce the learning rate and stop early if model is overfitting.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and repeated driving over more complex parts of track.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to come up with a model that is fast enough (real time) as well as results in as low as possible MSE.

As per the lecture videos I decided to use the NVIDIA model. I started with LeNet but it wasn't working very good and I was not able to reduce the MSE to the satisfiable measure. So I decided to use NVIDIA model.


![alt text][image1]

I trained my model with the combination of images that I generated + the images provided to us in the workspace. I split the data in the train and valid dataframes with 20% in the validation set and rest for the training. My initial training showed that my model was overfitting so I decided to use the L2 regularization in each of the layer along with the Dropout after the convolution layers. I also started with the high learning rate and used Keras callback `ReduceOnPlateau` to reduce learning rate if for 2 epochs `val_loss` didn't improve. I also used `EarlyStopping` if for 4 epochs `val_loss` didn't improve even after reducing `learning rate`. In my training data generator I have also used the combination of left, right and center images. I initially thought of using 3 different models for each type of images and finally ensembling them but that would have been overkill + slow. I used image augmentation to generate more data. I have used only the road side meaning I have cropped the images, `60px` from top and `25px` from bottom so that my model can focus on roads and doesn't distract with the noise such as trees or front bonet or car. I have used `random_brightness`, `random_flip` and `random_translation` for every randomly choosen images for left, right and center view. Also for each of the image I converted that image to `YUV` format. Becuase NVIDIA model works well with `YUV` format and also resized my images to `(66, 200, 3)`.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Below is the image attached of my NVIDIA architecure. 

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself. For example below is in example of image in which car is going from right to left. 

![alt text][image3]

To augment the data sat, I also flipped images and cropped the images.

After the collection process, I had close to 9k training data points. After using center + left + right and image augmentation training data points are close to 30k.


I finally randomly shuffled the data set and put 2k data points of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I started with 10 epochs. I used an adam optimizer with high learning rate `1e-2` and Keras callbacks to reduce learning rate if necessary.
