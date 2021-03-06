## Adapting MCF-Net to Predict Retinal Image Quality on a Continuous Scale
MCF-Net (short for Multiple Color-space Fusion Network) is a DL model used for predicting retinal image quality on a 3-level categorical scale, i.e. 'Good', 'Usable' and 'Reject', as detailed by [Fu et al. (2019)](https://github.com/HzFu/EyeQ). I have previously [re-implemented](https://github.com/fyii200/MCF_Net) the original model (made some changes to the code) so it can readily work with any datasets.  

### Motivation
The motivation behind turning the original MCF-Net into a (regression) model capable of predicting retinal image quality on a continuous scale arises from a project that aims to see if image quality filter threshold can be (and should be) treated as a tunable hyperparameter. Briefly, we hypothesise that there exists a Goldilocks level of image quality. And if it were to be used as a quality cutoff to filter the training set, the performance of a model (used in a given downstream task) on the unfiltered (i.e. no quality filter applied) test set would be optimal. Idenfying this optimal quality cutoff with a good degree of precision is conceivably contingent upon our ability to generate a continuous quality score for each image.  

## Re-implementation Steps
1. Download the trained model via this [link](https://uoe-my.sharepoint.com/:u:/g/personal/s2221899_ed_ac_uk/ESXnLxi8qzpJj4isMrTuzDMByQeB6FN4o6VFqqIZ-yHAJw?e=pkGwWN).
2. Keep the model in 'Regression_MCF_Net/MCF_Net'.
3. Retinal images are assumed to be found in 'Regression_MCF_Net/images'.
4. Run 'Regression_MCF_Net/MCF_Net/test_only.py' from terminal: python MCF_Net/test_only.py --result_name {name of result folder}
5. Result file will be saved to the home directory ('Regression_MCF_Net').

## Description of Adaptations
The original MCF-Net has 5 cross-entropy loss functions, i.e. 3 from the base networks (each deals with a particular colour space), 1 from the feature-level classification layer and 1 from the prediction-level classification layer (final prediction is made here). We removed the softmax function associated with each of these 5 loss functions and used mean absolute error (MAE) in place of cross entropy as the loss function. 

The final output is normalised between 0 (best) and 1 (worst). One important caveat is that the normalisation uses the minimum and maximum quality scores in a given dataset so the quality score really only represents the relative rank of each image in the dataset. As a simple but extreme example, in a dataset with only 2 images with very similar quality (but still slightly different), their normalised scores would be very different indeed (0 and 1, since each one of them has to be either minimum or maximum).

The adapted model was trained for 30 epochs (batch size = 4) using stohcastic gradient descent on the Eye-Quality training (n=12,543) dataset (original labels coded with 0, 0.5 and 1). Model from the training epoch that yielded the lowest MAE (28 in our case) on the validation set (n=4,875) was saved as the best model. The best model achieved an MAE of 0.15 on the test set (n=12,015).

## To do...
Validation (under construction)


