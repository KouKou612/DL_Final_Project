# Fake News/Images Detection
There are 3 models for fake news detection: CNN, logistic regression, and SVM

The dataset for fake news detection is from `https://www.kaggle.com/datasets/aadyasingh55/fake-news-classification?select=evaluation.csv`

There are 5 models for fake images detection: CNN, diffusion, gan, resnet, and SVM

The dataset for fake images detection is from `https`

## CNN_Fake_News
Install dependencies listed in `requirements.txt` in `CNN_Fake_News/` to run this part. 

Run `main.py` in `src/` to start training.

Result plots will be saved to the `src/`, see `output.txt`or a log during a previous training.

See `results` for plots for our previous training.


## LR+SVM_Fake_News.ipynb
This notebook implements text-based fake news detection using TF-IDF with Logistic Regression and Linear SVM.

Make sure the dataset files `train (2).csv` `evaluation.csv` and `test (1).csv` are in the same directory as the notebook

Run LR+SVM_Fake_News.ipynb to train and evaluate both models.

Model performance (accuracy, classification report, and confusion matrix) will be printed in the notebook output.

The notebook reports results on validation, evaluation, and test splits and serves as a baseline comparison for CNN-based fake news models.


## Fake_Images
Install dependencies listed in `requirements.txt` in the `Fake_Images/`  to run this part. 

Put the dataset folder in `Fake_Images/`, name it to `ddata`

Run `cnn.py`, `diffusion.py`, `gan.py`, `resnet.py`, `svm.py` to train corresponding model. The result plot will not be saved. 

See `results` for plots for our previous training.

### Image_DL.ipynb
This notebook contains the original end-to-end implementation for image-based fake image detection, including data loading, preprocessing, model training, and evaluation. It implements multiple approaches, including traditional machine learning baselines (LBP + SVM) and deep learning models such as a custom CNN, ResNet-based transfer learning, GAN-style discriminator models, and diffusion-inspired noisy CNNs. The notebook serves as the primary experimental prototype from which the standalone training scripts were later refactored. You can find further improvement in `cnn.py`, `diffusion.py`, `gan.py`, `resnet.py`, `svm.py`.


## Contribution 
Hao Huang: CNN_Fake_News, Revised `Image_DL.ipynb` to `cnn.py`, `diffusion.py`, `gan.py`, `resnet.py`, `svm.py`, `trainer.py`, training of all these models.

Guoqi Chen: Implemented the text-based fake news classification pipeline using TF-IDF with Logistic Regression and Linear SVM in `LR+SVM_Fake_News.ipynb`, identified the data and handled data preprocessing, model training, validation, testing, error analysis, and generating the reported metrics/results.

Ruiyang Wang: Using existing literature to identify models suitable for image-based fake news detection and implementing a basic image model as a foundation for future training.

Zhiyuan Li: Preprocessed the Fake Image dataset & Tested Image_DL.ipynb and gave out suggestions 





