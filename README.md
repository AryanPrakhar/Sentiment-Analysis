**Title: Sentiment Analysis of Text Data**

### Overview
This project focuses on sentiment analysis using the IMDB review dataset, consisting of 50,000 reviews labeled as positive or negative. The primary goal is to determine the sentiment of a given test review. The analysis involves creating a word cloud visualization of the most used words in positive and negative reviews and generating a horizontal bar chart of common words in positive reviews.

### Tools and Libraries
- Python
- NLTK for natural language processing
- Matplotlib and Seaborn for data visualization
- Plotly Express for interactive visualizations
- WordCloud for word frequency visualization
- Scikit-learn for machine learning tasks
- Pandas for data manipulation

### Dataset
- The dataset contains 50,000 entries with two columns: "review" (text) and "sentiment" (positive/negative).
- Initial exploration includes displaying dataset information, checking for duplicate entries, and visualizing the distribution of sentiments.

### Data Cleaning and Exploration
1. **Handling Missing Values**: No missing values are observed in the dataset.
2. **Univariate Analysis**: Utilizing count plots and distribution plots to understand the distribution of sentiments and reviewing sample reviews.

### Feature Engineering
1. **Word Count Analysis**: Creating a new feature for the number of words in each review.
2. **Text Preprocessing**: Lowercasing, removing stop words, URLs, special characters, and stemming.
3. **Word Cloud Visualization**: Visualizing the most frequent words in positive and negative reviews.
4. **Common Word Analysis**: Identifying and visualizing the most common words in positive and negative reviews.

### Modeling
1. **TF-IDF Vectorization**: Transforming text data into numerical format.
2. **Train-Test Split**: Splitting the dataset for training and testing.
3. **Machine Learning Models**: Training models like Logistic Regression, Naive Bayes, and Linear Support Vector Classifier (SVC).
4. **Model Evaluation**: Assessing models' performance using accuracy, confusion matrix, and classification reports.
5. **Hyperparameter Tuning**: Fine-tuning the Linear SVC model using GridSearchCV.

### Results
- **Linear SVC Model**: Achieved a test accuracy of 89.41%, outperforming Logistic Regression and Naive Bayes models.
- **Common Words Visualization**: Identified and visualized the most common words contributing to positive and negative sentiments.

### Future Work
- Fine-tuning other models for improved accuracy.
- Exploring deep learning models for sentiment analysis.
- Building a web application for user-friendly sentiment analysis.
