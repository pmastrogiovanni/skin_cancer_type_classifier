# Summary
I have built a multi-classifier on the HAM10000 dataset, which contains images of 7 types of skin cancer.
As classes are highly unbalanced, i performed data augmentation on the dataset.
My model is based on DenseNet121, allowing trainablity on some of its layers. Top was rebuilt employing multiple dropouts layers to address overfitting.
The model tested an accuracy of 85%.

You can download the model in the repository and load it in your notebook to test it out:

```
from keras.models import load_model

model = load_model('model.h5')
loss, accuracy = best_model.evaluate(test)
```


# Detailed description
## Original data

## Data augmentation

## The model

## Training

## Testing

## Potential Improvements

