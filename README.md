## Adapting MCF-Net to Predict Retinal Image Quality on a Continuous Scale
MCF-Net (short for Multiple Color-space Fusion Network) is a DL model used to predict retinal image quality on a 3-level categorical scale, i.e. 'Good', 'Usable' and 'Reject', as detailed in [Fu et al. (2019)](https://github.com/HzFu/EyeQ). I have previously [re-implemented](https://github.com/fyii200/MCF_Net) the original model so it can readily work with any datasets. 

PS: Regression MCF-Net is detailed in this [publication](https://link.springer.com/chapter/10.1007/978-3-031-16525-2_8).

### Re-implementation Steps
1. Download the trained model via this [link](https://uoe-my.sharepoint.com/:u:/g/personal/s2221899_ed_ac_uk/ESXnLxi8qzpJj4isMrTuzDMByQeB6FN4o6VFqqIZ-yHAJw?e=pkGwWN).
2. Save the model to 'Regression_MCF_Net/MCF_Net'.
3. Retinal images are expected to be fed from 'Regression_MCF_Net/images'.
4. Run 'Regression_MCF_Net/MCF_Net/test_only.py' from terminal: python MCF_Net/test_only.py --result_name {name of result folder}
5. Result file will be saved to the home directory ('Regression_MCF_Net').

### Description of Adaptations
The original MCF-Net has 5 cross-entropy loss functions, i.e. 3 from the base networks (each deals with a particular colour space), 1 from the feature-level classification layer and 1 from the prediction-level classification layer (final prediction is made here). We removed the softmax function associated with each of these 5 loss functions and used mean absolute error (MAE) in place of cross entropy as the loss function.

The final output is normalised between 0 (best) and 1 (worst). It is important to note that the normalisation uses the minimum and maximum quality scores specific to the given test dataset so the quality score really only represents the relative rank of each image in that dataset. 

The (adapted) model was retrained for 30 epochs (batch size = 4) using stohcastic gradient descent on the Eye-Quality training (n=12,543) dataset (original labels coded with 0, 0.5 and 1). Model from the training epoch that yielded the lowest MAE (28 in our case) on the validation set (n=4,875) was saved as the best model. The best model achieved an MAE of 0.15 on the test set (n=12,015).

## If you use any part of this work, please cite
```
@InProceedings{10.1007/978-3-031-16525-2_8,
author="Yii, Fabian SL and Dutt, Raman and MacGillivray, Tom and Dhillon, Baljean and Bernabeu, Miguel and Strang, Niall",
editor="Antony, Bhavna and Fu, Huazhu and Lee, Cecilia S. and MacGillivray, Tom and Xu, Yanwu and Zheng, Yalin",
title="Rethinking Retinal Image Quality: Treating Quality Threshold as a Tunable Hyperparameter",
booktitle="Ophthalmic Medical Image Analysis",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="73--83",
isbn="978-3-031-16525-2"
}
```

**Or** 

```
Yii, F.S., Dutt, R., MacGillivray, T., Dhillon, B., Bernabeu, M., Strang, N. (2022). Rethinking Retinal Image Quality: Treating Quality Threshold as a Tunable Hyperparameter. In: Antony, B., Fu, H., Lee, C.S., MacGillivray, T., Xu, Y., Zheng, Y. (eds) Ophthalmic Medical Image Analysis. OMIA 2022. Lecture Notes in Computer Science, vol 13576. Springer, Cham. https://doi.org/10.1007/978-3-031-16525-2_8
```

## And the original MCF-Net paper as appropriate
```
Huazhu Fu, Boyang Wang, Jianbing Shen, Shanshan Cui, Yanwu Xu, Jiang Liu, Ling Shao, "Evaluation of Retinal Image Quality Assessment Networks in Different Color-spaces", in MICCAI, 2019. [PDF] Note: The corrected accuracy score of MCF-Net is 0.8800.
```

