### Report on Naive Bayes Classifiers for Income Evaluation Dataset

*** Introduction:
This report explores the application of Naive Bayes classifiers to predict income levels based on demographic and employment data. The dataset used is 
"income_evaluation.csv".

*** Dataset Overview:

 *** Attributes:
  - The dataset includes various attributes such as `age`, `workclass`, `education`, `occupation`, `capital-gain`, `capital-loss`, `hours-per-week`, and
    `income` (target variable).

 *** Exploratory Data Analysis:
  - Initial exploration involved loading the dataset (`pd.read_csv()`), examining data types and missing values (`df.info()`), and statistical summaries (`df.describe()`).
  - Checked for class distribution in the target variable (`income`) using `value_counts()` and visualized it with a bar plot to understand the distribution of income levels.

*** Data Preprocessing:

 *** Handling Categorical Data:
  - Encoded the target variable (`income`) using `LabelEncoder` to convert categorical values into numeric format.
  - Applied one-hot encoding (`pd.get_dummies()`) to handle categorical features (`workclass`, `education`, `marital-status`, etc.) to prepare them for modeling.

 *** Data Splitting:
  - Split the dataset into training and testing sets (`train_test_split` from `sklearn.model_selection`) with a test size of 20% and a random state of 0 for
    reproducibility.

*** Naive Bayes Models:

*** Model-1: Gaussian Naive Bayes

  *** Description:**
   - Implemented Gaussian Naive Bayes (`GaussianNB`) suitable for continuous features assuming normally distributed data.
   - Trained the model on the training data (`x_train`, `y_train`) and predicted outcomes on the test set (`x_test`).

  ***Model Evaluation:
   - Calculated accuracy using `accuracy_score` and visualized the confusion matrix (`confusion_matrix`) using a heatmap (`sns.heatmap()`).

*** Model-2: Bernoulli Naive Bayes

  *** Description:
   - Utilized Bernoulli Naive Bayes (`BernoulliNB`) which is suitable for binary or boolean features.
   - Trained the model and evaluated performance metrics similar to Gaussian Naive Bayes.

*** Model-3: Multinomial Naive Bayes

  ***Description:
   - Applied Multinomial Naive Bayes (`MultinomialNB`) suitable for discrete features like word counts.
   - Trained the model, predicted outcomes, computed accuracy, and assessed performance with a confusion matrix.

*** Model Comparison:
  
  ***Performance Metrics:
   - Each Naive Bayes model (Gaussian, Bernoulli, Multinomial) demonstrated varying levels of accuracy and performance in predicting income levels.
   - **Gaussian Naive Bayes** assumed normal distribution and performed well on continuous data.
   - **Bernoulli Naive Bayes** handled binary features effectively and provided competitive accuracy.
   - **Multinomial Naive Bayes** was suitable for discrete features and achieved satisfactory predictive accuracy.

*** Conclusion:
- Naive Bayes classifiers are efficient and effective for categorical and numerical data classification tasks like income prediction.
- The choice of Naive Bayes model depends on the nature of the features and assumptions about the data distribution.
- Further model tuning, feature engineering, or exploring other ensemble techniques could potentially enhance predictive accuracy and robustness.

