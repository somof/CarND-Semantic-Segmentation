# Semantic Segmentation



## Rubric Points

### Build the Neural Network

#### Does the project load the pretrained vgg model?
This project loads the pre-trained VGG model at load_vgg().

#### Does the project learn the correct features from the images?
This project defines layers fot the FCN method at layers().

#### Does the project optimize the neural network?
This project defines its loss function and optimizer at optimize().

#### Does the project train the neural network?
This project trains correctly the network at train_nn().

### Neural Network Training

#### Does the project train the model correctly?
The model decreases loss over time on average as below.

![training curve](curve.png)


#### Does the project use reasonable hyperparameters?

- batch_size = 20
- other hyperparameters are same as Tensorflow initial values.
  - learning_rate = 0.001
- epoch number = 1000 (but loss < 0.01)
  - approximately 100 til the loss value became to be less than 0.01.

The batch size 20 is the value that shows the most smooth training-curve in some trial as below.

![batch size](batchsize.png)


#### Does the project correctly label the road?

The project works correctly, and labels the road.




add result


labels most pixels of roads close to the best solution.
  The model doesn't have to predict correctly all the images, just most of them.

- A solution that is close to best would label at least 80% of the
  road and label no more than 20% of non-road pixels as road.








## Set up

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
~/.pyenv/versions/3.5.4/bin/pip3 install tensorflow-gpu numpy scipy tqdm Pillow

install cuda-9.0
```

tensorflow-gpu==1.5.0
numpy==1.14.0
scipy==1.0.0
tqdm==4.19.5
opencv-python==3.4.0.12
Pillow==5.0.0



## FCN Model

## Dataset and Training Model

[Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) 

Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.




## Result

sample image


trainng curve



## CUDA ERROR

1,64,160,576 = 47022080 Bit  = 44.844 MB
4096,512,7,7 = 8.221e+08 Bit = 784 MB

2018-02-13 13:14:53.388304: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:895] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero




tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor with shape[1,64,160,576] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
	 [[Node: conv1_2/Conv2D = Conv2D[T=DT_FLOAT, data_format="NHWC", dilations=[1, 1, 1, 1], padding="SAME", strides=[1, 1, 1, 1], use_cudnn_on_gpu=true, _device="/job:localhost/replica:0/task:0/device:GPU:0"](conv1_1/Relu, conv1_2/filter)]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.

tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor with shape[1,64,160,576] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
	 [[Node: conv1_2/Conv2D = Conv2D[T=DT_FLOAT, data_format="NHWC", dilations=[1, 1, 1, 1], padding="SAME", strides=[1, 1, 1, 1], use_cudnn_on_gpu=true, _device="/job:localhost/replica:0/task:0/device:GPU:0"](conv1_1/Relu, conv1_2/filter)]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.



tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor with shape[4096,512,7,7] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
	 [[Node: fc6/Conv2D = Conv2D[T=DT_FLOAT, data_format="NHWC", dilations=[1, 1, 1, 1], padding="SAME", strides=[1, 1, 1, 1], use_cudnn_on_gpu=true, _device="/job:localhost/replica:0/task:0/device:GPU:0"](pool5, fc6/weights)]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.

	 [[Node: Softmax_20/_125 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_274_Softmax_20", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.





## The following is the original README provided by Udacity

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
