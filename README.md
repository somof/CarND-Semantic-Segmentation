# Semantic Segmentation



## Build the Neural Network

### Does the project load the pretrained vgg model?

- The function load_vgg is implemented correctly.

### Does the project learn the correct features from the images?

- The function layers is implemented correctly.

### Does the project optimize the neural network?

- The function optimize is implemented correctly.

### Does the project train the neural network?

- The function train_nn is implemented correctly.
  The loss of the network should be printed while the network is training.


## Neural Network Training

### Does the project train the model correctly?

- On average, the model decreases loss over time.

### Does the project use reasonable hyperparameters?

- The number of epoch and batch size are set to a reasonable number.

### Does the project correctly label the road?

- The project labels most pixels of roads close to the best solution.
  The model doesn't have to predict correctly all the images, just most of them.

- A solution that is close to best would label at least 80% of the
  road and label no more than 20% of non-road pixels as road.


## Create the spot instance in AWS

https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/614d4728-0fad-4c9d-a6c3-23227aef8f66/concepts/f6fccba8-0009-4d05-9356-fae428b6efb4#

View your EC2 Service Limit report at: https://console.aws.amazon.com/ec2/v2/home?#Limits

Find your "Current Limit" for the g2.2xlarge instance type.

Note: Not every AWS region supports GPU instances. If the region you've chosen does not support GPU instances, but you would like to use a GPU instance, then change your AWS region.


Submit a Limit Increase Request
From the EC2 Service Limits page, click on “Request limit increase” next to “g2.2xlarge”.
You will not be charged for requesting a limit increase. You will only be charged once you actually launch an instance.

We’ve created an AMI for you!
Search for the “udacity-carnd” AMI.
Select the g2.2xlarge instance type:
Increase the storage size to 16 GB (or more, if necessary):



How to run Semantic Segmentation on AWS ?
https://discussions.udacity.com/t/how-to-run-semantic-segmentation-on-aws/352069/61


Spot instances and AMI
https://discussions.udacity.com/t/spot-instances-and-ami/561250







## Pyenv setting

This project uses python 3.5.4 by pyenv,
and other after 3.3 versions of python would work as same as it.


sudo apt-get remove nvidia-*
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/375.66/NVIDIA-Linux-x86_64-375.66.run
sudo bash ./NVIDIA-Linux-x86_64-375.66.run  --dkms


```
# Amazon Linux
# yum -y install gcc gcc-c++ make git openssl-devel bzip2-devel zlib-devel readline-devel sqlite-devel
yum -y install git make

git clone https://github.com/yyuu/pyenv.git ~/.pyenv
or
cd ~/.pyenv; git pull

echo 'export PYENV_ROOT="${HOME}/.pyenv"' >> ~/.bashrc
echo 'if [ -d "${PYENV_ROOT}" ]; then' >> ~/.bashrc
echo 'export PATH=${PYENV_ROOT}/bin:$PATH' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'fi' >> ~/.bashrc
source ~/.bashrc

pyenv install 3.5.4
~/.pyenv/versions/3.5.4/bin/pip3 install tensorflow==1.4 numpy scipy tqdm Pillow
or
~/.pyenv/versions/3.5.4/bin/pip3 install tensorflow-gpu==1.4 numpy scipy tqdm Pillow
~/.pyenv/versions/3.5.4/bin/pip3 install tensorflow-gpu numpy scipy tqdm Pillow
```

~/.pyenv/versions/3.6.4/bin/pip3 install tensorflow numpy scipy tqdm Pillow opencv-python moviepy











## original README

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
