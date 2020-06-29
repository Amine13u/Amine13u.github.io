# A Dog Breed Classifier using Convolutional Neural Networks (CNN)  

![dogs](https://i.ytimg.com/vi/EKG89O05K5A/maxresdefault.jpg)

### Overview

If you are looking for a guided project related to deep learning and convolutional neural networks, this might be just it.

In this project, we will make the first steps towards developing an algorithm that could be used as part of a mobile or web app. At the end of this project, our code will accept any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling.

In this real-world setting, we will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed. There are many points of possible failure, and no perfect algorithm exists. Our imperfect solution will nonetheless create a fun user experience !

So let's begin !!!

## Part 0 : Import Datasets

### Import Dog Dataset

As a first step, we import a dataset of dog images. We populate a few variables through the use of the `load_files` function from the `scikit-learn` library:

* `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
* `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels
* `dog_names` - list of string-valued dog breed names for translating labels

### Import Human Dataset

Next, we import a dataset of human images, where the file paths are stored in the numpy array `human_files`

## Part 1 : Detect Humans

We use OpenCV's implementation of [Haar feature-based cascade classifiers](https://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images. OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).
Before using any of the face detectors, it is standard procedure to convert the images to grayscale. The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.

### Write a Human Face Detector

We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise. This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below :

            def face_detector(img_path):
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray)
                return len(faces) > 0
           
### Assess the Human Face Detector

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face. After testing we found :

    There are 100.0% of the first 100 images in human_files detected as human face
    There are 11.0% of the first 100 images in dog_files detected as human face

We see that our algorithm falls short of this goal, but still gives acceptable performance

## Part 2 : Detect Dogs

In this section, we use a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images. Given an image, this pre-trained ResNet-50 model returns a prediction for the object that is contained in the image.

### Pre-process the Data

When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape

                  (nb_samples,rows,columns,channels)
 
where `nb_samples` corresponds to the total number of images (or samples), and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively.
In this case, since we are working with color images, each image has three channels. Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths. It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds to a different image) in your dataset !

### Write a Dog Detector

While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), we notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from 'Chihuahua' to 'Mexican hairless'. Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the `ResNet50_predict_labels` function returns a value between 151 and 268 (inclusive).
We use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).

            def dog_detector(img_path):
                prediction = ResNet50_predict_labels(img_path)
                return ((prediction <= 268) & (prediction >= 151)) 

### Assess the Dog Detector

In testing the performance of our `dog_detector` function we found :

    There are 0.0% of the first 100 images in human_files detected as dog
    There are 100.0% of the first 100 images in dog_files detected as dog

Perfect !!

## Part 3 : Create a CNN to Classify Dog Breeds (using Transfer Learning)

Now let's use transfer learning to create a CNN using [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features that can identify dog breed from images. Our goal is to attain at least 60% accuracy on the test set.

### Obtain Bottleneck Features

We will extract the bottleneck features corresponding to the train, test, and validation sets by running the following code :

        bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
        train_Resnet50 = bottleneck_features['train']
        valid_Resnet50 = bottleneck_features['valid']
        test_Resnet50 = bottleneck_features['test']

### Model Architecture

The model uses the the pre-trained ResNet-50 model as a fixed feature extractor, where the last convolutional output of ResNet-50 is fed as input to our model. We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.

            _________________________________________________________________
            Layer (type)                 Output Shape              Param #   
            =================================================================
            global_average_pooling2d_2 ( (None, 512)               0         
            _________________________________________________________________
            dense_2 (Dense)              (None, 133)               68229     
            =================================================================
            Total params: 68,229
            Trainable params: 68,229
            Non-trainable params: 0
            _________________________________________________________________
            
### Test the Model

After compiling, training and loading the model with the best validation loss we can perform our test on the test dataset of dog images. And we did it, we got an accuracy of 80.6220% !!

Now let's predict Dog Breed with the model

### Predict Dog Breed with the Model

Our function should takes an image path as input and should returns the dog breed (Affenpinscher, Afghan_hound, etc) that is predicted by our model in three steps :

1. Extract the bottleneck features corresponding to the chosen CNN model
2. Supply the bottleneck features as input to the model to return the predicted vector. Note that the argmax of this prediction vector gives the index of the predicted dog breed
3. Use the `dog_names` array defined in Step 0 of this notebook to return the corresponding breed

        from extract_bottleneck_features import *
        def Resnet50_predict_breed(img_path):
            bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
            predicted_vector = Resnet50_model.predict(bottleneck_feature)
            return dog_names[np.argmax(predicted_vector)]

## Part 4 : Write the Algorithm

Now we will write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither. Then,

* if a dog is detected in the image, return the predicted breed.
* if a human is detected in the image, return the resembling dog breed.
* if neither is detected in the image, provide output that indicates an error.

For this, we will use the `face_detector`, `dog_detector`  and the `Resnet50_predict_breed` functions to predict dog breed.

The algorithm is as below :

    def image_classifier(img_path):
        img = cv2.imread(img_path)
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(cv_rgb)
        plt.show()
        
        if dog_detector(img_path):
            print('The predicted breed is {}'.format(Resnet50_predict_breed(img_path).split('.')[1]))
        elif face_detector(img_path):
            print('The resembling dog breed is {}'.format(Resnet50_predict_breed(img_path).split('.')[1]))          
        else:
            print('Error : Neither human or dog was detected in the image')

## Part 5 : Test Our Algorithm

Finnaly, we will test our algorithm on sample images !!

For this we will use six images and here we go :

![test1](<../images/test1.png>)
The predicted breed is Labrador_retriever

![test2](<../images/test2.png>)
The predicted breed is Brittany

![test3](<../images/test3.png>)
The resembling dog breed is Mastiff

![test4](<../images/test4.png>)
The predicted breed is American_eskimo_dog

![test5](<../images/test5.png>)
The resembling dog breed is Papillon

![test6](<../images/test6.png>)
Error : Neither human or dog was detected in the image

And we got 6 out of 6 correct predictions :)

## Part 6 : Conclusion

Even that we got 100% accuracy in our little test, that's don't mean that our algorithm is perfect for a simple reason that is six images is not a representative sample of test. 

For this reason, we provide  three potential points of improvement for it :

1.  We can train our model with a more complex data set.
2.  We can tune Hyper-parameters like weight initializing, learning rates, dropouts, batch_sizes, optimizers, ...
3.  We can use different model architectures to reduce the computational time while maintaining the accuracy.

Finally, you can  see more about this analysis, in the link to my [Github](https://github.com/Amine13u/Dog-Breed-Classifier) available here. 