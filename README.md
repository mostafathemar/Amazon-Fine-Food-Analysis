# Amazon Fine Food Reviews Analysis

This is a data science project that analyzes a dataset of food reviews from Amazon. The goal of the project is to predict the sentiment (positive/negative) of a review based on its text. The project involves data preprocessing, exploratory data analysis, feature extraction, model training, and model evaluation.

## Dataset

The dataset used in this project is the [Amazon Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews) dataset from Kaggle. The dataset consists of 568,454 food reviews Amazon users left up to October 2012.

## Technologies Used

- Python
- Pandas
- Scikit-Learn
- NLTK
- Numpy
- Matplotlib
- Seaborn

##  Data Loading and Preprocessing
The data for this project was loaded from the Amazon Fine Food Reviews dataset. The following preprocessing steps were applied:

1. Duplicate entries were removed.
2. HTML tags were stripped from the reviews.
3. Text was converted to lowercase and punctuation was removed.
4. Common English stopwords were removed.
5. The data was split into training and test sets.

# Featurization
1. TF-IDF
2. Word2Vec
3. Avg W2v
3. TF-IDF weighted W2v

## Model Training and Evaluation
Several machine learning models were trained on the data, including:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Random Forest Classifier
5. Gradient Boosting Classifier

Hyperparameter tuning was performed using RandomizedSearchCV to optimize the performance of the models. The models were then evaluated based on their accuracy, precision, recall, and F1-score. The best performing model was saved for future use.

## Conclusion and Next Steps
The best performing model was the RandomForest model with an accuracy of 99%.

Possible future improvements to this project include:

1. Experimenting with different vectorization techniques, such as Doc2Vec.
2. Trying more advanced models, like XGBoost or LightGBM.
3. Using a larger or more recent dataset for potentially better results.


## How to Run the Project

1. Clone this repository.
2. Download the dataset from Kaggle and put it in the same directory as the notebook.
3. Install the necessary Python packages. You can do this by running `pip install -r requirements.txt` (if you have pip installed) or `conda install --file requirements.txt` (if you're using the Anaconda distribution of Python).
4. Open the Jupyter notebook and run the cells to see the project in action.

## Project Structure

- `amazon_food_review_analysis.ipynb`: This Jupyter notebook contains all the code for the project.
- `Reviews.csv`: This is the Amazon Fine Food Reviews dataset. You'll need to download this from Kaggle.
- `README.md`: This README file provides information about the project.
- `requirements.txt`: This file lists the Python packages that you need to run the project.

## Acknowledgments

- Thanks to Kaggle for the dataset.
