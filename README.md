# Summary
I have built a multi-classifier on the HAM10000 dataset, which contains images of 7 types of skin cancer.

Due to the significant class imbalance, I applied data augmentation to the dataset.

My model is based on DenseNet121, allowing trainablity on some of its layers. To tackle overfitting, I reconstructed the top by incorporating multiple dropout layers.

The model tested an accuracy of 85% 
(Training was performed on a Macbook pro M3 GPU).

You can download the model in the repository and load it in your notebook to test it out:

```python
from keras.models import load_model

model = load_model('model.h5')
loss, accuracy = best_model.evaluate(test)
```


# Detailed description
Below you can find a detailed decription of each step of the project, including motivations and code snippets.
To access full code, refer to the jupyter notebook in the repository.

## Original data

## Data augmentation

## The model

## Training

## Testing

## Potential Improvements

- Enhanced data augmentation -> Employing a broader range of diverse and realistic augmentations could facilitate the model in generalizing more effectively and efficiently. To achieve this, one might consider incorporating a Generative Adversarial Network.
- Hyperparameter tuning -> Due to resource constraints, a comprehensive grid search for fine-tuning hyperparameters (dropout rate, learning rate, batch size, decay factor, number of frozen layers) was not feasible. Utilizing parallelization could offer a potential solution.
- Altered architecture -> Employing a model with reduced depth and complexity might potentially aid in mitigating overfitting. Further testing would be necessary to assess its effectiveness.

