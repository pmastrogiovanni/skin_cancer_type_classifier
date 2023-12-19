# Summary
I have built a multi-classifier on the HAM10000 dataset, which contains images of 7 types of skin cancer.  

Due to the significant class imbalances, I applied data augmentation to the dataset.  

My model is based on DenseNet121, allowing trainablity on some of its layers. To tackle overfitting, I reconstructed the top by incorporating multiple dropout layers.  

The model tested an accuracy of 84% 
(Training was performed on a Macbook pro M3 GPU).  

You can download the model in the repository and load it in your notebook to test it out:

```python
from keras.models import load_model

model = load_model('model.h5')
loss, accuracy = best_model.evaluate(test)
```

<br>

# Detailed description
Below you can find a detailed decription of each step of the project, including motivations and code snippets.
To access full code, refer to the jupyter notebook in the repository.

### Original data
The dataset is composed of rouglhy 10.000 images of skin lesions, with an highly unbalanced distribution between classes, as highlighted by the plot:  






### Data augmentation
Since classes are unbalanced, I decided to perform data augmentation on the less represented classes (with less than 1000 images) on the training set.  

I have built an Augmenter object, which creates 3 modified images from each image in the selected classes.  

The applied modifications are:
- Rotation --> randomly between -90 and 90 degrees
- Flip --> with a 50% probability
- Blur --> addition of random intensity of gaussian blur
- Contrast --> random contrast between 0.4 (darker) and 1.4 (lighter)

The augmentation is process is completely reproducible thanks to seeds.

The class distribution is now as follows:



### Data Preparation
To decrease memory consumption, dataset is pre-processed and loaded in batches of 32 images at each iteration.  

Pre-process includes:
- Resising to shape 256,256,3 (RGB channels).
- Rescaling values between 0 and 1.
- Storing labels into one hot encoded vectors.

Dataset was split into 70% training, 20% validation and 10% test.

### The model
As the dataset is relatively small, I decided to use a pre-trained model to allow for a better performance. As shown in the benchmark form ..., DenseNet121 is the pre-trained model that works better for this task. I allowed for retraining of the model layers except for the feature extraction layers, as they are used to detect low-level features in images. I have rebuilt the top of the model by introducing multiple dropout layers to address overfitting.

```python
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
for layer in base_model.layers[:149]:
    layer.trainable = False
for layer in base_model.layers[149:]:
    layer.trainable = True

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.7))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))
```

### Training

The model was trained using Adam as optimizer, with a learning rate of 0.001. To address overfitting, a learning scheduler with 0.7 as decay rate was employed.

Best model was saved by using early stopping whenever the validation accuracy was not increasing for 3 epochs.




### Testing

### Potential Improvements

- Enhanced data augmentation -> Employing a broader range of diverse and realistic augmentations could facilitate the model in generalizing more effectively and efficiently. To achieve this, one might consider incorporating a Generative Adversarial Network.
- Hyperparameter tuning -> Due to resource constraints, a comprehensive grid search for fine-tuning hyperparameters (dropout rate, learning rate, batch size, decay factor, number of frozen layers) was not feasible. Utilizing parallelization could offer a potential solution.
- Altered architecture -> Employing a model with reduced depth and complexity might potentially aid in mitigating overfitting. Further testing would be necessary to assess its effectiveness.

