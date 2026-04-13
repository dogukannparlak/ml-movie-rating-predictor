# Predicting Personal Movie Ratings Using Machine Learning: A Comparative Study of Classification and Regression Approaches

---

## Abstract

Predicting subjective movie ratings from objective metadata remains a challenging task due to the inherently personal nature of viewer preferences. This study investigates the feasibility of predicting personal IMDb ratings on a 1–10 scale using a dataset of 1,312 movies characterized by 22 features, including runtime, release year, and 20 binary genre indicators. The problem is formulated under both regression and classification paradigms, with four distinct classification schemes (binary, three-class balanced, three-class strict, and four-class) and one continuous regression target. A total of seven classification algorithms (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, Support Vector Machine, Multilayer Perceptron, and a Stacking ensemble) and five regression algorithms (Random Forest, Gradient Boosting, Support Vector Regression, XGBoost, and Multilayer Perceptron) are systematically evaluated using hold-out validation (80/20 split) and stratified five-fold cross-validation. Three feature selection strategies—filter, wrapper, and embedded methods—are compared against a full-feature baseline. Results indicate that Gradient Boosting achieves the highest binary classification performance with an accuracy of 75.3% and an ROC-AUC of 0.809, while Support Vector Regression attains the lowest mean absolute error of 0.934 with an R² of 0.299 in the regression task. The Horror genre, Runtime, and Drama emerge as the most influential predictors across both tasks. Feature selection does not yield consistent improvements over the baseline with 22 features, suggesting that the original feature set is already compact and informative. These findings demonstrate that binary classification of personal ratings is moderately feasible from metadata alone, whereas fine-grained multi-class prediction and precise regression remain limited by the absence of richer contextual features.

---

## 1. Introduction

The proliferation of digital streaming platforms and online movie databases has generated vast quantities of user-generated rating data, creating both opportunities and challenges for computational approaches to preference modeling. Understanding and predicting individual movie preferences has become a central concern in recommender systems research, with applications spanning personalized content delivery, audience segmentation, and cultural analytics. While collaborative filtering approaches leverage inter-user similarity patterns, content-based methods attempt to predict ratings from intrinsic properties of the items themselves, offering the advantage of operating without requiring data from other users.

This study addresses the problem of predicting a single user's personal movie ratings from readily available movie metadata. The dataset under investigation comprises 1,312 movies drawn from a personal IMDb ratings export, where each movie is characterized by its runtime in minutes, release year, and membership in 20 genre categories encoded as binary indicator variables. The target variable, "Your Rating," is recorded on a discrete 1–10 integer scale. Descriptive analysis reveals that the ratings exhibit a unimodal distribution concentrated around values of 5 and 6, which together account for 56.18% of all observations, with a mean rating of 5.38 and a standard deviation of 1.45. This distribution reflects a natural tendency toward moderate ratings, with relatively few extreme scores at either end of the scale.

The prediction task is formulated under two complementary paradigms. In the regression formulation, the goal is to predict the raw numeric rating on the continuous 1–10 scale. In the classification formulation, the numeric ratings are discretized into categorical labels using four distinct binning strategies: a binary split (low versus high), a quantile-based three-class balanced partition, a domain-informed three-class strict partition, and a quantile-based four-class partition. This multi-formulation approach enables a systematic comparison of how different levels of target granularity affect predictive performance and provides insights into the trade-off between prediction specificity and accuracy.

The methodological framework follows the Cross-Industry Standard Process for Data Mining (CRISP-DM), encompassing data understanding, data preparation, feature engineering, feature selection, modeling, and evaluation. A diverse set of machine learning algorithms is employed, including both individual models and ensemble methods. Three feature selection strategies—filter-based, wrapper-based, and embedded methods—are systematically compared to assess whether dimensionality reduction improves predictive performance on this relatively low-dimensional dataset. All experiments are conducted with rigorous evaluation protocols, including stratified hold-out validation and five-fold cross-validation, with preprocessing steps encapsulated within scikit-learn pipelines to prevent data leakage.

The principal contributions of this study are fourfold. First, it provides a systematic comparison of regression and multiple classification formulations for personal rating prediction from identical feature sets. Second, it evaluates a comprehensive suite of seven classification and five regression algorithms under consistent experimental conditions. Third, it investigates the utility of three distinct feature selection paradigms on a compact, metadata-only feature space. Fourth, it identifies the most influential movie attributes for personal preference prediction through multiple feature importance methodologies, including correlation analysis, mutual information, recursive feature elimination, tree-based importances, and permutation importance.

---

## 2. Literature Review

The prediction of movie ratings has been extensively studied within the broader context of recommender systems and preference modeling. This section reviews the relevant literature across several dimensions: recommender system paradigms, rating prediction methodologies, ensemble learning, feature selection, and evaluation frameworks.

Recommender systems are generally categorized into collaborative filtering, content-based filtering, and hybrid approaches (Ricci et al., 2015). Collaborative filtering methods, which predict user preferences based on the preferences of similar users, gained prominence through the Netflix Prize competition (Bennett & Lanning, 2007). Matrix factorization techniques, as formalized by Koren et al. (2009), became the dominant approach for collaborative filtering by decomposing the user-item interaction matrix into latent factor representations. However, collaborative filtering suffers from the cold-start problem and requires multi-user data, limitations that motivate content-based approaches when only single-user data is available.

Content-based filtering predicts user preferences by modeling the relationship between item attributes and user ratings (Lops et al., 2011). Early content-based systems relied on simple feature matching, while more recent approaches employ machine learning models to learn complex attribute-preference mappings. Pazzani and Billsus (2007) provided a comprehensive overview of content-based recommendation techniques, demonstrating that even basic metadata features such as genre can provide meaningful predictive signal. The present study aligns with this content-based paradigm, as it relies exclusively on movie metadata to predict ratings.

The formulation of rating prediction as a machine learning task has been approached through both regression and classification lenses. Regression approaches aim to predict the exact numeric rating, while classification approaches discretize ratings into categorical bins (Diao et al., 2014). The choice between these formulations involves a trade-off between granularity and accuracy, as classification typically achieves higher nominal accuracy at the cost of reduced prediction specificity. Several studies have investigated this trade-off, finding that binary or coarse-grained classification often outperforms fine-grained prediction when feature sets are limited (Ahmed et al., 2015).

Ensemble learning methods have demonstrated superior performance across a wide range of prediction tasks. Random Forests, introduced by Breiman (2001), construct collections of decorrelated decision trees through bagging and random feature subsampling, achieving robust performance with minimal hyperparameter tuning. Gradient Boosting Machines, formalized by Friedman (2001), build sequential ensembles where each successive model corrects the errors of its predecessors, offering strong predictive performance particularly for tabular data. Chen and Guestrin (2016) introduced XGBoost, an optimized implementation of gradient boosting that incorporates regularization and efficient computation, achieving state-of-the-art results in numerous machine learning competitions. Stacking ensembles, as described by Wolpert (1992), combine predictions from multiple base learners through a meta-learner, potentially capturing complementary patterns from diverse model architectures.

Support Vector Machines (SVMs), introduced by Cortes and Vapnik (1995), have been widely applied to both classification and regression tasks. The kernel trick enables SVMs to capture non-linear decision boundaries in high-dimensional feature spaces, and the formulation as a convex optimization problem guarantees convergence to a global optimum. Support Vector Regression (SVR) extends the SVM framework to continuous prediction by defining an epsilon-insensitive loss function (Drucker et al., 1997). Multilayer Perceptrons (MLPs), as representative neural network architectures, offer flexible function approximation capabilities and have shown competitive performance on structured data when properly regularized (Goodfellow et al., 2016).

Feature selection is a critical preprocessing step that aims to identify the most informative subset of features, potentially improving model performance, reducing computational cost, and enhancing interpretability (Guyon & Elisseeff, 2003). Feature selection methods are broadly categorized into filter, wrapper, and embedded approaches. Filter methods evaluate features independently of the learning algorithm using statistical measures such as correlation coefficients and mutual information (Vergara & Estévez, 2014). Wrapper methods, such as Recursive Feature Elimination (RFE), evaluate feature subsets based on model performance, offering potentially higher accuracy at greater computational cost (Guyon et al., 2002). Embedded methods integrate feature selection into the model training process, with tree-based feature importances being a prominent example (Strobl et al., 2007). The comparative effectiveness of these methods depends on dataset characteristics, and no single method consistently dominates (Chandrashekar & Sahin, 2014).

The evaluation of predictive models requires careful selection of appropriate metrics and validation strategies. For classification tasks, accuracy alone can be misleading in the presence of class imbalance, motivating the use of F1-score, precision, recall, and area under the ROC curve (AUC) as complementary metrics (Sokolova & Lapalme, 2009). For regression tasks, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (R²) provide different perspectives on prediction quality (Willmott & Matsuura, 2005). Cross-validation, particularly stratified k-fold cross-validation for classification tasks, provides more reliable performance estimates than single hold-out splits, especially on smaller datasets (Kohavi, 1995). The use of scikit-learn pipelines to encapsulate preprocessing and modeling steps prevents data leakage, a common pitfall that can lead to overly optimistic performance estimates (Pedregosa et al., 2011).

The CRISP-DM (Cross-Industry Standard Process for Data Mining) framework provides a structured methodology for organizing data mining projects (Wirth & Hipp, 2000). Its six phases—business understanding, data understanding, data preparation, modeling, evaluation, and deployment—offer a systematic approach that has been widely adopted in both academic and industrial settings (Mariscal et al., 2010). The present study follows the CRISP-DM framework to structure the experimental pipeline, ensuring methodological rigor and reproducibility.

Class imbalance is a pervasive challenge in classification tasks where the distribution of class labels is skewed (He & Garcia, 2009). When certain classes are underrepresented, classifiers tend to be biased toward the majority class, resulting in poor performance on minority classes. Strategies for addressing class imbalance include algorithmic approaches such as cost-sensitive learning and class-weight adjustment, as well as data-level approaches such as oversampling and undersampling (Chawla et al., 2002). The use of balanced class weights in algorithms such as Logistic Regression and Random Forest, as employed in the present study, represents a commonly used algorithmic mitigation strategy.

The application of machine learning to movie rating prediction using metadata features has been explored in several prior studies. Nithin et al. (2014) applied multiple regression and classification algorithms to IMDb data, finding that genre and runtime were among the most predictive metadata features. Similarly, Dhir and Raj (2018) demonstrated that ensemble methods outperform individual classifiers for movie rating prediction tasks. The present study extends this line of research by providing a comprehensive comparison across multiple classification granularities, regression, and feature selection strategies within a unified experimental framework.

---

## 3. Methodology

The experimental methodology follows the CRISP-DM framework, proceeding through business understanding, data understanding, data preparation, feature engineering, feature selection, modeling, and evaluation. All experiments are implemented in Python using scikit-learn, with a fixed random state of 42 to ensure reproducibility.

### 3.1 Business Understanding

The objective of this study is to develop machine learning models capable of predicting a user's personal movie rating based on readily available movie metadata. Such predictive models have practical applications in personalized content recommendation, where a system could suggest movies likely to receive high ratings from a specific individual based on their historical preferences. The business goal is therefore to understand which movie attributes most strongly influence personal rating behavior and to determine which modeling approach—regression or classification at various granularity levels—offers the most effective prediction framework.

### 3.2 Data Understanding

The dataset consists of 1,312 movies extracted from a personal IMDb ratings export. Each record includes the movie's original title, the user's personal rating on a 1–10 integer scale, runtime in minutes, release year, a comma-separated genre string, and 20 binary genre indicator columns derived from the genre string. The 22 features used for modeling comprise two continuous variables (Runtime in minutes and Year) and 20 binary genre indicators (Action, Adventure, Animation, Biography, Comedy, Crime, Drama, Family, Fantasy, History, Horror, Music, Musical, Mystery, Romance, Sci-Fi, Sport, Thriller, War, and Western).

Descriptive statistics reveal that the mean runtime is 109.66 minutes (SD = 21.34), ranging from 75 to 242 minutes, and the mean release year is 2007.75 (SD = 11.37), spanning from 1922 to 2026. Among the genre indicators, Thriller (46.1%), Horror (37.3%), Action (32.5%), Drama (31.4%), and Comedy (30.2%) are the most prevalent categories, while Music (1.2%), Sport (1.2%), Western (1.1%), and History (1.6%) are the least represented.

The target variable, "Your Rating," has a mean of 5.38 (SD = 1.45) and exhibits a unimodal distribution concentrated around the middle of the scale. As presented in Figure 1, the most frequent ratings are 6 (29.88%, n = 392) and 5 (26.30%, n = 345), followed by 7 (14.33%, n = 188) and 4 (13.72%, n = 180). The extreme ratings of 1 (0.76%, n = 10), 9 (0.76%, n = 10), and 10 (0.15%, n = 2) are rare, indicating that the user tends to assign moderate ratings and seldom uses the extremes of the scale.

*Figure 1. Distribution of personal ratings across 1,312 movies (see `outputs/figures/rating_histogram.png`).*

Correlation analysis between features and the target variable reveals that the Horror genre exhibits the strongest negative correlation (r = −0.362), indicating that horror movies tend to receive lower ratings. Drama shows the strongest positive correlation (r = 0.349), followed by Runtime (r = 0.316) and Crime (r = 0.212). The release Year shows negligible linear correlation with rating (r = 0.002). The full correlation structure among all features and the target is presented in Figure 2, which reveals notable inter-feature correlations such as between Animation and Family (r = 0.79) and between Action and Adventure (r = 0.36).

*Figure 2. Pearson correlation heatmap for all features and the target variable (see `outputs/figures/correlation_heatmap.png`).*

A missing value analysis confirms that no missing values exist in any feature or in the target variable, as all 22 features and the target report zero missing counts across all 1,312 observations.

### 3.3 Data Preparation

Data preparation involves missing value handling, target variable construction, and train-test splitting. Although no missing values are present in the current dataset, the preprocessing pipeline includes provisions for median imputation of continuous features and zero-fill imputation of binary genre indicators to ensure robustness against potential missing data in future deployments.

The target variable is prepared under five distinct formulations. For regression, the raw "Your Rating" value (1–10) serves directly as a continuous target. For classification, four binning strategies are applied. The binary formulation assigns ratings below 6 to the "low" class and ratings of 6 or above to the "high" class, producing a nearly balanced split of 657 low-rated movies (50.1%) and 655 high-rated movies (49.9%). The three-class balanced formulation uses quantile-based binning to create approximately equal-frequency classes, resulting in Class 0 (657 samples, 50.1%), Class 1 (392 samples, 29.9%), and Class 2 (263 samples, 20.0%), with bin edges at rating values of 1.0, 5.0, 6.0, and 10.0. The three-class strict formulation applies domain-informed thresholds, defining low (ratings 1–4, 312 samples, 23.8%), medium (ratings 5–6, 737 samples, 56.2%), and high (ratings 7–10, 263 samples, 20.0%). The four-class formulation also applies quantile-based binning; however, due to the distributional properties of the ratings, the quantile procedure produces three effective bins identical to the three-class balanced formulation.

*Figure 3. Class distributions for all four classification formulations (see `outputs/figures/class_distribution_binary.png`, `class_distribution_3class_balanced.png`, `class_distribution_3class_strict.png`, `class_distribution_4class.png`).*

The dataset is partitioned into training (80%) and test (20%) sets using a fixed random state of 42. For classification tasks, stratified splitting is employed to preserve class proportions in both partitions.

### 3.4 Feature Engineering

The feature set consists of 22 variables extracted from the original IMDb data. The two continuous features, Runtime (mins) and Year, capture temporal and structural properties of the movies. The 20 binary genre indicator variables are derived from the original comma-separated "Genres" column through one-hot encoding, where each genre column takes a value of 1 if the movie belongs to that genre and 0 otherwise. Since movies may belong to multiple genres simultaneously, these indicators are not mutually exclusive. No additional feature transformations, polynomial features, or interaction terms are constructed, as the goal is to assess how much predictive information is contained in these basic metadata attributes.

### 3.5 Feature Selection

Three feature selection strategies are applied and compared against a full-feature baseline to determine whether dimensionality reduction improves predictive performance on this 22-feature dataset.

The filter method combines two statistical criteria: absolute Pearson correlation with the target variable (threshold of 0.05) and mutual information ranking (top 15 features). Features passing either criterion are retained in the selected set through a union operation. For the classification target, the filter method selects the majority of features, as most genre indicators exceed the minimal correlation threshold.

The wrapper method employs Recursive Feature Elimination (RFE) using Logistic Regression as the estimator for classification tasks and Random Forest for regression tasks. RFE iteratively removes the least important features based on model coefficients or importances, retaining 10 features in the final subset. For the classification task, the wrapper method selects Runtime, Action, Biography, History, Horror, Drama, Sci-Fi, Fantasy, Animation, and Adventure, as indicated by their RFE ranking of 1 in the feature selection results.

The embedded method leverages tree-based feature importance scores from Random Forest and XGBoost models. The importance scores from both models are averaged, and the top 12 features are retained. For classification, the embedded method identifies Runtime (importance = 0.160), Year (0.133), Horror (0.087), Biography (0.075), Drama (0.067), Sci-Fi (0.053), Action (0.050), Mystery (0.037), Crime (0.034), Romance (0.034), Fantasy (0.031), and History (0.031) as the most informative features.

A consensus analysis reveals that eight features are selected by all three methods for the classification task: Runtime, Action, Biography, History, Horror, Drama, Sci-Fi, and Fantasy. Notably, Musical is the only feature not selected by any of the three methods. For the regression task, nine features achieve unanimous selection across all three methods: Runtime, Year, Action, Adventure, Drama, Comedy, Sci-Fi, Fantasy, and Horror.

*Figure 4. Embedded feature importance rankings for classification and regression tasks (see `outputs/figures/feature_importance_embedded_classification.png`, `feature_importance_embedded_regression.png`).*

### 3.6 Modeling

Seven classification algorithms are evaluated. Logistic Regression is configured with balanced class weights and a maximum of 1,000 iterations, with feature scaling applied through a StandardScaler within the pipeline. Random Forest is configured with 200 trees and balanced class weights. Gradient Boosting uses 200 estimators with a learning rate of 0.1 and maximum depth of 3. XGBoost is configured with the multi-class softprob objective and a verbosity level of 0. Support Vector Machine employs an RBF kernel with probability estimation enabled, balanced class weights, and pipeline-integrated standard scaling. The Multilayer Perceptron architecture comprises two hidden layers of 64 and 32 neurons, using the Adam optimizer with early stopping. The Stacking classifier combines Random Forest, Gradient Boosting, and SVM as base estimators with Logistic Regression as the meta-learner, using cross-validated predictions for training the meta-model.

Five regression algorithms are evaluated. Random Forest and Gradient Boosting regressors are configured analogously to their classification counterparts, with light hyperparameter tuning via GridSearchCV optimizing for negative mean absolute error. Support Vector Regression uses an RBF kernel with standard scaling applied through a pipeline. XGBoost regression uses the squared error objective. The MLP regressor uses the same architecture as its classification counterpart with early stopping enabled.

For both Random Forest and Gradient Boosting, light hyperparameter tuning is conducted using GridSearchCV with three-fold cross-validation, optimizing macro F1-score for classification and negative MAE for regression. The tuned parameters include the number of estimators and maximum depth.

All models are trained on four feature sets: the full baseline set of 22 features, the filter-selected subset, the wrapper-selected subset, and the embedded-selected subset. This results in a total of 28 classification experiments per task (7 models × 4 feature sets) and 20 regression experiments (5 models × 4 feature sets).

### 3.7 Evaluation Strategy

Model evaluation is conducted through two complementary protocols. The primary evaluation uses a hold-out strategy with an 80/20 train-test split (stratified for classification tasks), where models are trained on the training partition and evaluated on the held-out test set. The secondary evaluation employs five-fold cross-validation (stratified for classification, standard for regression) to obtain more robust performance estimates with associated variance measures.

For classification tasks, the following metrics are computed: accuracy, macro-averaged F1-score, weighted-averaged F1-score, macro-averaged precision, macro-averaged recall, log loss, and ROC-AUC (for binary classification only). The macro-averaged F1-score serves as the primary metric for model selection, as it equally weights performance across all classes regardless of their frequency. For the regression task, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (R²) are computed, with MAE serving as the primary selection criterion.

---

## 4. Results

This section presents the experimental results for each prediction task, beginning with the binary classification, proceeding through the multi-class formulations, and concluding with the regression analysis. All reported metrics are computed on the hold-out test set unless otherwise specified. Cross-validation results are provided for additional validation.

### 4.1 Binary Classification

The binary classification task distinguishes between low-rated (rating < 6) and high-rated (rating ≥ 6) movies. Table 1 presents the hold-out test performance of all seven models on the baseline (full) feature set.

**Table 1.** Binary classification results on the hold-out test set (baseline features).

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | Precision (Macro) | Recall (Macro) | Log Loss | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.741 | 0.741 | 0.741 | 0.743 | 0.741 | 0.541 | 0.799 |
| Random Forest | 0.722 | 0.722 | 0.722 | 0.724 | 0.722 | 0.526 | 0.814 |
| Gradient Boosting | 0.753 | 0.752 | 0.752 | 0.755 | 0.753 | 0.534 | 0.809 |
| XGBoost | 0.715 | 0.715 | 0.715 | 0.715 | 0.715 | 0.559 | 0.799 |
| SVM | 0.722 | 0.722 | 0.722 | 0.723 | 0.722 | 0.557 | 0.793 |
| MLP | 0.703 | 0.703 | 0.703 | 0.705 | 0.703 | 0.547 | 0.784 |
| Stacking | 0.738 | 0.737 | 0.737 | 0.739 | 0.738 | 0.533 | 0.813 |

Gradient Boosting achieves the highest accuracy (0.753) and macro F1-score (0.752) among all models, followed closely by Logistic Regression (accuracy = 0.741, F1 = 0.741) and the Stacking ensemble (accuracy = 0.738, F1 = 0.737). Notably, the Stacking ensemble, despite combining Random Forest, Gradient Boosting, and SVM as base learners, does not surpass Gradient Boosting as an individual model. Random Forest achieves the highest ROC-AUC (0.814), suggesting superior ranking ability, though its classification accuracy (0.722) is lower than that of Gradient Boosting. MLP achieves the lowest performance across all metrics, with an accuracy of 0.703 and an ROC-AUC of 0.784.

The confusion matrix for the best-performing Gradient Boosting model (Figure 5) reveals that the model correctly classifies 105 out of 132 low-rated movies and 93 out of 131 high-rated movies on the test set. The model produces 27 false positives (low-rated movies predicted as high) and 38 false negatives (high-rated movies predicted as low), indicating a slight bias toward predicting the low class. The ROC curve (Figure 6) demonstrates solid discriminative performance with an AUC of 0.81, with the curve exhibiting a favorable trade-off between true positive rate and false positive rate across most operating thresholds.

*Figure 5. Confusion matrix for Gradient Boosting on the binary classification task (see `outputs/figures/cm_binary_GradientBoosting.png`).*

*Figure 6. ROC curve for Gradient Boosting on the binary classification task (see `outputs/figures/roc_binary_GradientBoosting.png`).*

The impact of feature selection on binary classification performance is examined through the results across four feature sets. For the best-performing Gradient Boosting model, the baseline achieves 0.753 accuracy, the filter-selected subset achieves 0.715, the wrapper-selected subset achieves 0.715, and the embedded-selected subset achieves 0.692. This pattern—where the baseline consistently outperforms all feature-selected subsets—is observed across the majority of models, suggesting that the original 22-feature set is already sufficiently compact and that removing features leads to information loss without compensating computational or generalization benefits.

### 4.2 Three-Class Balanced Classification

The three-class balanced task uses quantile-based binning to create approximately equally distributed classes. Table 2 presents the baseline results.

**Table 2.** Three-class balanced classification results on the hold-out test set (baseline features).

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | Precision (Macro) | Recall (Macro) | Log Loss |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.586 | 0.549 | 0.585 | 0.544 | 0.558 | 0.895 |
| Random Forest | 0.631 | 0.598 | 0.626 | 0.597 | 0.601 | 0.837 |
| Gradient Boosting | 0.627 | 0.571 | 0.612 | 0.614 | 0.557 | 0.863 |
| XGBoost | 0.597 | 0.550 | 0.591 | 0.565 | 0.542 | 0.999 |
| SVM | 0.563 | 0.540 | 0.565 | 0.532 | 0.556 | 0.850 |
| MLP | 0.570 | 0.469 | 0.527 | 0.519 | 0.472 | 0.941 |
| Stacking | 0.589 | 0.551 | 0.588 | 0.545 | 0.561 | 0.878 |

Random Forest achieves the highest performance with an accuracy of 0.631 and a macro F1-score of 0.598, outperforming Gradient Boosting (accuracy = 0.627, F1 = 0.571) in this task. The discrepancy between accuracy and macro F1-score for several models, particularly MLP (accuracy = 0.570, F1 = 0.469), indicates uneven performance across the three classes. This is further evidenced by the confusion matrix (Figure 7), which shows that Random Forest correctly classifies 100 out of 132 samples in Class 0 (the largest class), but only 33 out of 78 in Class 1 and 33 out of 53 in Class 2. The model tends to over-predict Class 0 at the expense of the smaller classes.

*Figure 7. Confusion matrix for Random Forest on the three-class balanced classification task (see `outputs/figures/cm_3class_balanced_RandomForest.png`).*

### 4.3 Three-Class Strict Classification

The three-class strict formulation uses domain-informed thresholds: low (ratings 1–4), medium (ratings 5–6), and high (ratings 7–10). This produces a notably imbalanced distribution, with the medium class comprising 56.2% of the data. Table 3 presents the baseline results.

**Table 3.** Three-class strict classification results on the hold-out test set (baseline features).

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | Precision (Macro) | Recall (Macro) | Log Loss |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.532 | 0.533 | 0.525 | 0.541 | 0.605 | 0.900 |
| Random Forest | 0.593 | 0.569 | 0.595 | 0.568 | 0.573 | 0.822 |
| Gradient Boosting | 0.665 | 0.587 | 0.644 | 0.672 | 0.558 | 0.773 |
| XGBoost | 0.608 | 0.547 | 0.599 | 0.571 | 0.534 | 0.936 |
| SVM | 0.532 | 0.536 | 0.526 | 0.539 | 0.603 | 0.789 |
| MLP | 0.620 | 0.554 | 0.607 | 0.592 | 0.536 | 0.818 |
| Stacking | 0.544 | 0.543 | 0.541 | 0.534 | 0.594 | 0.880 |

Gradient Boosting achieves the highest accuracy (0.665) and macro F1-score (0.587), with notably higher precision (0.672) than recall (0.558). The confusion matrix (Figure 8) reveals that Gradient Boosting correctly classifies 128 out of 148 medium-class samples but only 28 out of 62 low-class and 19 out of 53 high-class samples. This indicates that the model's overall accuracy is substantially inflated by its strong performance on the dominant medium class, while it struggles considerably with the minority low and high classes. The model misclassifies 33 of 62 low-class samples and 33 of 53 high-class samples as medium, demonstrating a systematic bias toward the majority class despite the relatively high overall accuracy.

*Figure 8. Confusion matrix for Gradient Boosting on the three-class strict classification task (see `outputs/figures/cm_3class_strict_GradientBoosting.png`).*

### 4.4 Four-Class Classification

The four-class formulation, designed to use quantile-based binning with four target bins, produces results identical to the three-class balanced formulation. This occurs because the quantile-based binning procedure, when applied to the rating distribution concentrated at values 5 and 6, yields only three distinct bin edges ([1.0, 5.0, 6.0, 10.0]), effectively collapsing the intended four classes into three. Consequently, the four-class and three-class balanced tasks produce identical results, with Random Forest achieving the best performance (accuracy = 0.631, F1 macro = 0.598). This outcome highlights the sensitivity of quantile-based binning to the shape of the target distribution and underscores the importance of examining the actual bin structure rather than assuming that the specified number of bins will be realized.

### 4.5 Regression Results

The regression task predicts the raw "Your Rating" value on the 1–10 continuous scale. Table 4 presents the hold-out test performance for all five regression models on the baseline feature set.

**Table 4.** Regression results on the hold-out test set (baseline features).

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 0.984 | 1.275 | 0.240 |
| Gradient Boosting | 0.950 | 1.229 | 0.294 |
| SVR | 0.937 | 1.241 | 0.280 |
| XGBoost | 1.083 | 1.397 | 0.088 |
| MLP | 0.971 | 1.266 | 0.251 |

On the hold-out test set, SVR achieves the lowest MAE (0.937) and Gradient Boosting achieves the highest R² (0.294), while XGBoost performs substantially worse than all other models with an MAE of 1.083 and an R² of 0.088. Notably, the best overall regression result across all feature sets is achieved by SVR with the filter-selected feature subset, yielding an MAE of 0.934, RMSE of 1.224, and R² of 0.299—the only instance where feature selection produces a marginal improvement over the baseline.

**Table 5.** Cross-validation regression results (baseline features, 5-fold).

| Model | MAE (Mean ± SD) | RMSE (Mean ± SD) | R² (Mean ± SD) |
|---|---|---|---|
| Random Forest | 0.936 ± 0.025 | 1.205 ± 0.040 | 0.303 ± 0.051 |
| Gradient Boosting | 0.921 ± 0.023 | 1.193 ± 0.022 | 0.318 ± 0.033 |
| SVR | 0.901 ± 0.022 | 1.182 ± 0.036 | 0.331 ± 0.036 |
| XGBoost | 1.001 ± 0.042 | 1.287 ± 0.066 | 0.205 ± 0.080 |
| MLP | 0.957 ± 0.026 | 1.229 ± 0.033 | 0.277 ± 0.029 |

The five-fold cross-validation results (Table 5) confirm SVR as the best-performing regression model, achieving the lowest mean MAE (0.901 ± 0.022) and the highest mean R² (0.331 ± 0.036). Gradient Boosting ranks second with a mean MAE of 0.921 ± 0.023 and a mean R² of 0.318 ± 0.033. SVR also exhibits relatively low variance across folds, indicating stable performance. XGBoost again performs worst, with the highest MAE (1.001 ± 0.042) and lowest R² (0.205 ± 0.080), and shows the highest standard deviation, suggesting instability.

The predicted-versus-actual scatter plot for the best-performing SVR model (Figure 9) reveals that predictions cluster heavily in the 4–6 range, regardless of the actual rating. The model correctly tracks the general trend for mid-range ratings but substantially underestimates high ratings (8–10) and overestimates low ratings (1–2), a pattern consistent with regression toward the mean. The residual plot (Figure 10) shows a systematic pattern where residuals increase for extreme actual values, confirming that the model struggles to predict ratings far from the mean of 5.38.

*Figure 9. Predicted versus actual ratings for SVR on the regression task (see `outputs/figures/reg_pred_vs_actual_SVR.png`).*

*Figure 10. Residual plot for SVR on the regression task (see `outputs/figures/reg_residuals_SVR.png`).*

### 4.6 Feature Importance Analysis

Multiple feature importance methods consistently identify a core set of highly predictive features. Table 6 summarizes the top five features according to each importance methodology.

**Table 6.** Top five features by importance methodology.

| Rank | Correlation (abs.) | Embedded (Clf.) | Embedded (Reg.) | Permutation (Clf.) | Permutation (Reg.) |
|---|---|---|---|---|---|
| 1 | Horror (0.362) | Runtime (0.160) | Horror (0.186) | Horror | Runtime |
| 2 | Drama (0.349) | Year (0.133) | Runtime (0.152) | Runtime | Horror |
| 3 | Runtime (0.316) | Horror (0.087) | Year (0.124) | Animation | Sci-Fi |
| 4 | Crime (0.212) | Biography (0.075) | Drama (0.117) | Action | Animation |
| 5 | Biography (0.177) | Drama (0.067) | Sci-Fi (0.051) | Sci-Fi | Action |

Across all methods, Horror and Runtime consistently emerge as the two most important predictors. Horror's strong negative correlation (r = −0.362) with the target indicates that the user systematically rates horror movies lower. Runtime's positive correlation (r = 0.316) suggests a preference for longer films, which may reflect a confound with movie quality or production budget. Drama's positive association (r = 0.349) indicates a genre preference for dramatic content. The Year feature, while showing negligible linear correlation (r = 0.002), ranks highly in tree-based importance measures, suggesting it captures non-linear or interaction effects that linear correlation fails to detect.

*Figure 11. Permutation importance for the best binary classification model (see `outputs/figures/permutation_importance_classification.png`).*

*Figure 12. Permutation importance for the best regression model (see `outputs/figures/permutation_importance_regression.png`).*

### 4.7 Feature Selection Impact

Table 7 summarizes the performance of the best model for each task across all four feature sets.

**Table 7.** Best model performance across feature sets.

| Task | Model | Baseline | Filter | Wrapper | Embedded |
|---|---|---|---|---|---|
| Binary (Acc.) | Gradient Boosting | 0.753 | 0.715 | 0.715 | 0.692 |
| 3-Class Bal. (Acc.) | Random Forest | 0.631 | 0.608 | 0.521 | 0.578 |
| 3-Class Strict (Acc.) | Gradient Boosting | 0.665 | 0.639 | 0.608 | 0.627 |
| Regression (MAE) | SVR | 0.937 | 0.934 | 0.958 | 0.946 |

For classification tasks, the baseline feature set consistently produces the highest performance. The filter method yields the smallest degradation, while the wrapper method produces the largest performance drop, particularly for the three-class balanced task where Random Forest accuracy falls from 0.631 to 0.521. For regression, the filter method produces a marginal improvement for SVR (MAE from 0.937 to 0.934), but all other feature-selection and model combinations produce worse results than their respective baselines.

---

## 5. Discussion

The experimental results reveal several noteworthy patterns regarding the predictability of personal movie ratings from metadata, the relative effectiveness of different modeling approaches, and the role of feature selection in a compact feature space.

The superiority of Gradient Boosting in the binary and three-class strict classification tasks can be attributed to its sequential error-correction mechanism, where each successive tree is trained to correct the residual errors of the ensemble. This approach is particularly effective when the decision boundary is complex and non-linear, as is likely the case when distinguishing between subjective rating categories based on genre and runtime features. The model's ability to capture non-linear interactions between features—such as the interaction between Horror genre membership and Runtime—likely contributes to its advantage over Logistic Regression, which assumes a linear relationship between features and the log-odds of class membership. Random Forest's stronger performance on the three-class balanced task may reflect its robustness to noisy labels and its effective use of balanced class weights, which compensate for the unequal class sizes (50.1%, 29.9%, 20.0%) more effectively than Gradient Boosting's sequential approach.

The consistently poor performance of XGBoost across both classification and regression tasks is noteworthy, as XGBoost typically matches or exceeds standard Gradient Boosting in many benchmarks. The high log loss values observed for XGBoost (e.g., 0.559 for binary classification versus 0.534 for Gradient Boosting) suggest poor probability calibration, potentially resulting from its default regularization settings being suboptimal for this particular dataset size and feature configuration. The limited hyperparameter tuning applied in this study—restricted to the number of estimators and maximum depth for Random Forest and Gradient Boosting only—may have disadvantaged XGBoost, which has a larger hyperparameter space including learning rate, regularization parameters, and subsampling ratios.

The Stacking ensemble, despite combining three diverse base learners (Random Forest, Gradient Boosting, and SVM), fails to outperform Gradient Boosting as an individual model in any classification task. This outcome suggests that the base learners do not capture sufficiently complementary patterns to benefit from meta-learning. Given the limited feature space (22 features, predominantly binary), the diversity among base learners may be constrained by the shared information content of the features, reducing the potential for stacking to discover novel patterns.

The feature importance analysis provides interpretable insights into the user's rating behavior. The strong negative influence of the Horror genre across all importance methods indicates a consistent personal aversion to horror content, which systematically depresses ratings regardless of other movie attributes. The positive influence of Runtime suggests that the user tends to rate longer movies more favorably, a pattern that may reflect a preference for epic narratives, prestige productions, or simply greater engagement with extended storytelling formats. The positive influence of Drama aligns with a preference for narrative-driven content with emotional depth. The emergence of Year as an important feature in tree-based methods (despite near-zero linear correlation) suggests temporal patterns in rating behavior that manifest through non-linear or threshold effects—for instance, the user may rate movies from certain decades differently, a pattern that linear correlation cannot capture.

The failure of feature selection to improve performance—and indeed its tendency to degrade performance—is a significant finding that warrants discussion. With only 22 features, the feature space is already compact relative to the dataset size (1,312 observations), yielding a favorable observation-to-feature ratio of approximately 60:1. Under these conditions, the risk of overfitting due to high dimensionality is minimal, and removing features is more likely to discard predictive information than to reduce noise. This finding is consistent with theoretical considerations suggesting that feature selection is most beneficial when the feature space is large relative to the sample size or when many features are irrelevant (Guyon & Elisseeff, 2003). The marginal improvement of the filter method for SVR in the regression task (MAE improving from 0.937 to 0.934) may represent a small reduction in noise features that benefits the kernel-based learner, though the difference is unlikely to be statistically significant.

The class imbalance in the three-class strict formulation exerts a pronounced effect on model behavior. With 56.2% of observations belonging to the medium class (ratings 5–6), classifiers face a strong prior incentive to predict the majority class. The confusion matrix for Gradient Boosting (Figure 8) vividly illustrates this effect: 128 of 148 medium-class samples are correctly classified, but only 28 of 62 low-class and 19 of 53 high-class samples are correctly identified. Despite the use of balanced class weights in applicable models, the imbalance still manifests in prediction patterns, particularly for models like Gradient Boosting that do not directly incorporate class-weight adjustments. The gap between accuracy (0.665) and macro F1-score (0.587) in this task quantifies the extent to which overall accuracy overstates the model's ability to identify minority classes.

The comparison between classification and regression paradigms reveals a fundamental trade-off between prediction specificity and achievable accuracy. Binary classification attains a reasonable accuracy of 75.3%, but it provides only a coarse determination of whether a movie will be rated above or below average. Moving to three classes reduces accuracy to approximately 63–67%, and the R² of 0.299 for the best regression model indicates that only about 30% of the variance in personal ratings is explained by metadata features. This outcome is not unexpected, as personal movie ratings are influenced by numerous factors beyond basic metadata, including plot quality, performances, directorial style, personal mood, prior expectations, and cultural context—none of which are captured in the present feature set.

The regression model's tendency to predict values clustered around the mean (as shown in the predicted-versus-actual scatter plot) is a manifestation of regression to the mean, exacerbated by the limited predictive power of the features. When the features provide only modest predictive signal, the optimal prediction under squared error loss shifts toward the population mean, resulting in compressed predictions that underestimate the true variance of ratings. This effect is particularly pronounced for extreme ratings, where the model systematically underpredicts high values and overpredicts low values.

---

## 6. Conclusion

This study has presented a comprehensive investigation of personal movie rating prediction using machine learning, comparing regression and multiple classification formulations on a dataset of 1,312 movies characterized by 22 metadata features. The findings support several concluding observations.

Binary classification of personal ratings into low (below 6) and high (6 and above) categories is moderately feasible using Gradient Boosting, achieving an accuracy of 75.3%, a macro F1-score of 0.752, and an ROC-AUC of 0.809 on the hold-out test set. This level of performance, while not sufficient for precise recommendation, demonstrates that metadata features carry meaningful signal about personal preferences.

Multi-class classification performance degrades as the number of classes increases, with three-class balanced accuracy reaching 63.1% (Random Forest) and three-class strict accuracy reaching 66.5% (Gradient Boosting). The higher accuracy in the strict formulation, despite greater class imbalance, is partly attributable to the model's ability to exploit the dominant medium class, as evidenced by the disparity between accuracy and macro F1-score.

Regression using SVR achieves a mean absolute error below one rating point (MAE = 0.934 on the best hold-out configuration, confirmed at 0.901 in cross-validation), indicating that predictions are, on average, less than one point away from the true rating on the 1–10 scale. However, the R² of 0.299 reveals that approximately 70% of the variance in personal ratings remains unexplained by the metadata features employed.

The Horror genre, Runtime, and Drama genre consistently emerge as the most influential predictors across all importance methodologies, providing interpretable insights into the user's rating preferences: a systematic aversion to horror content, a preference for longer movies, and a preference for dramatic narratives.

Feature selection does not provide consistent improvements over the baseline 22-feature set, as the feature space is already compact and informative for the given dataset size. The baseline configuration is therefore recommended as the default for this prediction task.

---

## 7. Limitations

Several limitations constrain the generalizability and interpretability of the findings reported in this study.

The dataset comprises ratings from a single user, reflecting one individual's subjective preferences. As a result, the learned models are user-specific and cannot be assumed to generalize to other viewers. The identified feature importances (e.g., the negative influence of Horror) represent personal preferences rather than universal patterns, and a different user might exhibit entirely different feature-preference relationships.

The feature set is restricted to basic metadata attributes—runtime, release year, and genre indicators. This representation excludes numerous potentially informative features such as director identity, cast members, budget, plot summaries, user reviews, visual style, soundtrack characteristics, and critical reception scores. The relatively low R² of 0.299 in the regression task suggests that these excluded features likely account for a substantial portion of the unexplained variance.

The dataset contains 1,312 observations, which, while sufficient for training the relatively low-complexity models employed, may be insufficient to capture subtle patterns or to reliably estimate the performance of more complex models. The note in the experiment documentation regarding result sensitivity to different splits reflects this concern.

The target distribution is heavily concentrated around ratings of 5 and 6 (56.18% combined), with very few extreme ratings (ratings of 1, 9, and 10 together comprise only 1.67% of the data). This distributional skew limits the ability of any model to accurately predict extreme ratings and introduces inherent imbalance into classification formulations.

Temporal dynamics in user preferences are not modeled. The dataset spans movies from 1922 to 2026, and it is plausible that the user's preferences have evolved over time. The static modeling approach treats all ratings as exchangeable, potentially missing temporal trends in preference patterns.

---

## 8. Future Work

Several directions for future research emerge from the limitations and findings of the present study.

Incorporating natural language processing features derived from plot summaries, user reviews, or movie descriptions could substantially enrich the feature space and improve predictive performance. Sentiment analysis of reviews and topic modeling of plot descriptions may capture aspects of movie content that genre labels and runtime cannot represent.

Adding structural features such as director identity, principal cast members, production company, and budget information could provide additional predictive signal, as user preferences often extend beyond genre to encompass stylistic and qualitative factors associated with specific filmmakers and actors.

Extending the framework to a multi-user setting using collaborative filtering or hybrid recommender approaches would enable the models to leverage inter-user similarity patterns. Matrix factorization or neural collaborative filtering methods could potentially achieve higher prediction accuracy by borrowing strength from like-minded users.

Addressing class imbalance through data-level techniques such as SMOTE (Synthetic Minority Over-sampling Technique) or ADASYN could improve multi-class classification performance, particularly for underrepresented classes. Additionally, exploring cost-sensitive learning frameworks that assign asymmetric misclassification penalties may better align model optimization with the practical goal of accurate minority-class identification.

Applying deep learning approaches, such as deep neural networks with entity embeddings for categorical features, or attention-based architectures that can model feature interactions explicitly, may capture complex non-linear patterns that traditional machine learning methods miss.

Modeling temporal dynamics through time-aware features or recurrent architectures could capture evolving user preferences, enabling the system to give greater weight to recent rating behavior when predicting future preferences.

---

## 9. Acknowledgements

The authors acknowledge the CSE315 Machine Learning course at the Department of Computer Science and Engineering for providing the academic framework and motivation for this study. The scikit-learn, XGBoost, and matplotlib libraries were instrumental in the implementation and evaluation of the experimental pipeline.

---

## 10. References

Ahmed, M., Afzal, H., Siddiqi, I., Amjad, M. F., & Khurshid, K. (2015). Exploring nested ensemble learners using overproduction and choose approach for churn prediction in telecom industry. *Neural Computing and Applications*, 27(8), 2241–2251. https://doi.org/10.1007/s00521-015-2068-2

Bennett, J., & Lanning, S. (2007). The Netflix Prize. *Proceedings of KDD Cup and Workshop*, 2007. https://www.cs.uic.edu/~liub/Netflix-KDD-Cup-2007.pdf

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chandrashekar, G., & Sahin, F. (2014). A survey on feature selection methods. *Computers & Electrical Engineering*, 40(1), 16–28. https://doi.org/10.1016/j.compeleceng.2013.11.024

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357. https://doi.org/10.1613/jair.953

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297. https://doi.org/10.1007/BF00994018

Dhir, R., & Raj, A. (2018). Movie success prediction using machine learning algorithms and their comparison. *Proceedings of the 1st International Conference on Secure Cyber Computing and Communication*, 385–390. https://doi.org/10.1109/ICSCCC.2018.8703320

Diao, Q., Qiu, M., Wu, C.-Y., Smola, A. J., Jiang, J., & Wang, C. (2014). Jointly modeling aspects, ratings and sentiments for movie recommendation. *Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 193–202. https://doi.org/10.1145/2623330.2623758

Drucker, H., Burges, C. J. C., Kaufman, L., Smola, A., & Vapnik, V. (1997). Support vector regression machines. *Advances in Neural Information Processing Systems*, 9, 155–161.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157–1182.

Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. *Machine Learning*, 46(1–3), 389–422. https://doi.org/10.1023/A:1012487302797

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239

Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *Proceedings of the 14th International Joint Conference on Artificial Intelligence*, 2, 1137–1145.

Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30–37. https://doi.org/10.1109/MC.2009.263

Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based recommender systems: State of the art and trends. In F. Ricci, L. Rokach, B. Shapira, & P. B. Kantor (Eds.), *Recommender systems handbook* (pp. 73–105). Springer. https://doi.org/10.1007/978-0-387-85820-3_3

Mariscal, G., Marbán, Ó., & Fernández, C. (2010). A survey of data mining and knowledge discovery process models and methodologies. *The Knowledge Engineering Review*, 25(2), 137–166. https://doi.org/10.1017/S0269888910000032

Nithin, V. R., Pranav, M., Sarath Babu, P. B., & Lijiya, A. (2014). Predicting movie success based on IMDb data. *International Journal of Data Mining Techniques and Applications*, 3(2), 365–368.

Pazzani, M. J., & Billsus, D. (2007). Content-based recommendation systems. In P. Brusilovsky, A. Kobsa, & W. Nejdl (Eds.), *The adaptive web* (pp. 325–341). Springer. https://doi.org/10.1007/978-3-540-72079-9_10

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender systems: Introduction and challenges. In F. Ricci, L. Rokach, & B. Shapira (Eds.), *Recommender systems handbook* (2nd ed., pp. 1–34). Springer. https://doi.org/10.1007/978-1-4899-7637-6_1

Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002

Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). Bias in random forest variable importance measures: Illustrations, sources and a solution. *BMC Bioinformatics*, 8, 25. https://doi.org/10.1186/1471-2105-8-25

Vergara, J. R., & Estévez, P. A. (2014). A review of feature selection methods based on mutual information. *Neural Computing and Applications*, 24(1), 175–186. https://doi.org/10.1007/s00521-013-1368-0

Willmott, C. J., & Matsuura, K. (2005). Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance. *Climate Research*, 30(1), 79–82. https://doi.org/10.3354/cr030079

Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. *Proceedings of the 4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining*, 29–39.

Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259. https://doi.org/10.1016/S0893-6080(05)80023-1
