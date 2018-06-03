### convolutional neural nets architecture
#### convnet from scratch
shallow cnn (softmax activation vs. squared-hinge-loss function with l2-regulation) and then replace the top softmax layer or squared-hinge-loss function with SVM which implemented in sklearn.svm.SVC, finally hit 62.3% accuracy on fer2013 PrivateTest dataset.

#### pre-trained VGG16 convolutional base
First, pre-trained your new stochastic initial fully-connected layer with frozen convolutional base, then unfrozen the top convolutional block to fine-tuning the convolutional base in order to extract better features, Finally the output of flatten layer feed into L2-SVM hit 65.47% accuracy on fer2013 PrivateTest dataset.

### conclusion
#### data augment?
I also use the data augment with keras.preprocessing.image.ImageDataGenerator, however performance did not improve, Maybe you can explore deeper. If you get a better performance, please share with me, I'll appreciate it.
#### cnn vs. dnn
As you can see, the features extracted with dnn(fine-tuning VGG16 nets) is better than shallow cnn which is implemented from scratch, but former is so expensive that you should attempt it if you have access to a GPU. If you just want to inspect the performance, the process will be very fast.
#### softmax vs. SVM
If you run the above code, i believe you already get it. SVM multi-classifier clearly precedes the softmax activation function, at least in my experiment.

### Reference:
[Deep learning with Python](https://www.manning.com/books/deep-learning-with-python)

[Deep Learning using Linear Support Vector Machines](http://deeplearning.net/wp-content/uploads/2013/03/dlsvm.pdf): the winner(hit 71.2% accuracy) in [Facial emotion recognition in Kaggle competition](https://github.com/zlpure/Facial-Expression-Recognition)
