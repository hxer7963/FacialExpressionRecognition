### Usage
You can find the fer2013 dataset in [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) uncompress it to dataset directory which you should create at the top of the project. Then run the Utils. to convert the sequential pixels value to images which stored in the dataset directory. If you are confuse about directory hierarchy, you can feel free to inspect the source code in Utils.py. Furthermore, if you have any trouble in extracting the image pixels value to dataset directory, you can feel free to download my processed [dataset](https://pan.baidu.com/s/1ijrbx2FgoBaN71rvPO6S2w), which power by Baidu cloud disk.

If you just want to inspect the model which I have trained, you can download the [Models](https://pan.baidu.com/s/1GthSB0k0vJbtgyuQF3kNDQ) from Baidu cloud disk which also have no password.

### convolutional neural nets architecture
#### convnet from scratch
shallow cnn (softmax activation) and then replace the top softmax layer with SVM multiple-classifier which implemented in sklearn.svm.SVC, finally hit 62.3% accuracy on PrivateTest dataset.

#### pre-trained VGG16 convolutional base
First, pre-trained your new stochastic initial fully-connected layer with **frozen** convolutional base, then unfrozen the top convolutional block to fine-tuning the convolutional base in order to extract better features, Finally the output of flatten layer feed into L2-SVM hit 65.47% accuracy on PrivateTest dataset.

### conclusion
#### cnn vs. dnn
As you can see, the features extracted with dnn(fine-tuning VGG16 nets) is better than shallow cnn which is implemented from scratch, but former is so expensive that you should attempt it if you have access to a GPU. If you just want to inspect the performance of processed model, the process will be very fast.
#### softmax vs. SVM
If you run the above code, i believe you already get it. SVM multi-classifier clearly precedes the softmax activation function, at least in my experiment.

### Reference:
[Deep learning with Python](https://www.manning.com/books/deep-learning-with-python)

[Deep Learning using Linear Support Vector Machines](http://deeplearning.net/wp-content/uploads/2013/03/dlsvm.pdf): the winner(hit 71.2% accuracy) in [Facial emotion recognition in Kaggle competition](https://github.com/zlpure/Facial-Expression-Recognition)
