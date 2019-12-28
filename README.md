# Google Landmark Recognition Challenge

- Kaggle Competition Page: [Google Landmark Recognition 2019](https://www.kaggle.com/c/landmark-recognition-2019)
- Applied VGG-16 and ResNet-50 with data augmentation
- Achieved a classification accuracy of 93.77% and 95.34%, respectively.

## Preprocessing

- `preprocessing.ipynb`
    - Select and sampling data
    - Output two files:
        - `train_200.csv`: To increase the model stabil- ity and accelerate the training process, only the images with its corresponding landmark occurrence more than 200 are selected for modeling.
            - The total numbers of images are 630277, 88887, 88887 corresponding to train- ing, validating and testing set. The total number of training labels (i.e. the classes that need to be recognized) is 1066.
        - `train_sample_temp.csv`: used for testing and debugging. Only selected classes containing 1000 images and then randomly sampled images from these classes.
- `seperate_files`:
    - Applied stratified shuffle split to seperate data into
        - training (78% for model training): 630277 images
        - validation (11% for hype-parameters tuning): 88887 images
        - testing (11%, for final evaluation): 88887 images
    - based on the metadata of images in `train_200.csv`, split 'train directory' into training, validation and testing directories using python shutil library
    - Final Directory Tree
    ![directory](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/fa8c8514-6a8d-45b4-b5ab-828f213a4a41/Screen_Shot_2019-11-29_at_4.14.03_PM.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAT73L2G45ICXGZGC4%2F20191129%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20191129T083916Z&X-Amz-Expires=86400&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEH8aCXVzLXdlc3QtMiJHMEUCIEWvHI5AhKURJRd0wVVgKtvYD1nDn6UWhD1%2FTaqK7xZWAiEArTaKNp4epXMHQBfwE9bmvxVnXm3wgGRQZpCTqk2FQ%2Fsq2wIIuP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgwyNzQ1NjcxNDkzNzAiDGDErH9Jn2QKNmL7qiqvAkQtWQJJOp1DSkWlstfuY%2BUDspsIGO5G73SdgklMHmP7VMj5EYZz4Zk8dRtlW2kczCQ734SrUu%2FLpOdKZf8Uiiic7icgCT9CS9N%2B4Xc90U810vNx53wgi8sHmMb6X0QugTdpeS%2FF6unz7chYkoCKtoV%2FxlHLVYcG52zCmEsrjX5ciVUv%2FgwdP6UNC1UUdwlwPYG49VdTI8ZnenOXqBUUpJEY0naG9cvCApMjsfcLbmc170MU0hPwKWmrV99uUD32baY63t2VTt220kdwR8ASerYrtA8mol7w%2FASipyH0%2FAiebQkZbIIpe7k8IIHREWDXogBrKg4A6NAMWLlzFtneLX8oWxLboFmwXbN%2FJQQyrP7njvzR0bhrJeifLEbeOMztXZz4xoxZFCfn3dxTW%2B3mdzDXgoPvBTrNAgZdadYJX%2BoKHpMTM10ByEuxDZ3u6UbkqsAayGBUCCDY25DFEAOZE%2BBK43UN%2FJXH35pZyDDnXRlsBynhljzeX7dTmV81fNVrWXODYKxZyI7BQoIXta0meS%2BUuQgR1v9MT5uFRqrgdO9%2B%2F9KJQaItPApPEMURmNOSZGma6Eb6XwbX7HW0H%2Fsiu%2FMMTF25x%2Fd2ZLLeK4%2FSWeUgr076Lxjp%2Fzw8Hm3pW4UqZvn%2BjWQOV0Mddx8cQveV2nwsDtn5a0shVcJ%2BuCyAm7wV7QzVqoxIUElQoILOanzABJjO6AsGz5SWHFkynqWehv1bQ%2BhdzNuG8HLzGP5fAlrhkVlOWXSsXC4ku2y28T6%2BkCjFmb8BePdJk4NI3N8ETRdGCir6eHijxBna2rxIWxC9QAao15rJsUKmgovwugnq%2B1ivzL%2F3DYRM1IQPcVlTT1U4jZ0m1g%3D%3D&X-Amz-Signature=25beecb238ea165dcf4423f8c1ec945c924175ae5d08ce29fbaedeb2a714edf1&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Screen_Shot_2019-11-29_at_4.14.03_PM.png%22)

# Modeling

- `modeling_stage1.ipynb`:
    - started with a relatively small dataset and fed the dataset into LeNet-5, a very basic and light- weighted ConNet.
    - Images are randomly sampled from each class and loaded batches of those images during train- ing by a custom data generator (detailed python script in `generator.py`)
    - The purpose for this stage is to assure that:
        - all the methods in the pipeline were used correctly.
        - size of images (128x128x3) corresponds with the size set in the input layer
        - the loading process is correct
- `modeling_stage2.ipynb`: Training without Data Augmentation
    - we created a imageDataGenerator object without setting or passing any parameters, which means we were going to use original images for training
    - We built up using two pre-trained models in this stage, VGG-16 and ResNet-50. Since the input size of our data is different from the data in ImageNet, we did not include the top layers in VGG or ResNet.
    - VGG-16
        - we added Global Average Pooling Layer and a fully connected layer with softmax as the activation function at the end
        - Hyper-parameters
            - number of epochs: 8
            - batch size: 128
            - optimizer: Adam with learning rate = 0.0001
            - Top two layers frozen
    - ResNet-50
        - only the final fully connected layer with softmax is added as the activation function in the final layer.
        - Hyper-parameters
            - number of epochs: 8
            - batch size: 128
            - optimizer: Adam
            - learning rate
                - initialized at 0.0001
                - halved at 5th epoch
                - halved again at 7th epoch
            - All layers trainable
- `data_augmentation.ipynb`
    - All the parameters of these transformations were passed into ImageDataGenerator object. With flow_ from_directory, ImageDataGenerator loaded the data from specific directory and perform real-time data augmentation during training.
        - rotation
        - width shift
        - height shift
        - zoom
    - Only use ResNet-50
    - Hyper-parameters:
        - batch size: 128
        - optimizer: adam
        - learning rate: initialized as 1e-4 with step decay
        - all layers trainable

# Environment setup

- Google Cloud Platform
- 8-CPUs, 30 GB virtual machine
- NVIDIA Tesla 100P GPU.
- The average training time for all the models in stage 2 and 3 is around 3.5 hours for total 8 epochs.
