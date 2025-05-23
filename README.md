# Fake News Detection with Python and Machine Learning 🕵️

A machine learning project that uses Natural Language Processing (NLP) techniques to automatically classify news articles as REAL or FAKE with 92.82% accuracy.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

In today's digital age, fake news has become a significant concern, spreading rapidly through social media and online platforms. This project implements a machine learning solution to automatically detect and classify news articles as authentic or fabricated.

The model uses **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization combined with a **Passive Aggressive Classifier** to achieve high accuracy in distinguishing between real and fake news articles.

## ✨ Features

- **High Accuracy**: Achieves 92.82% accuracy on test data
- **NLP Processing**: Uses TF-IDF vectorization for text feature extraction
- **Fast Classification**: Efficient online learning algorithm (Passive Aggressive Classifier)
- **Robust Preprocessing**: Handles stop words and document frequency filtering
- **Detailed Analysis**: Provides confusion matrix and performance metrics

## 📊 Dataset

The project uses a news dataset containing:
- **Articles**: News article text content
- **Titles**: Headlines of the news articles
- **Labels**: Binary classification (REAL/FAKE)
- **Format**: CSV file with preprocessed news data

### Dataset Structure
```
- Unnamed: 0: Index column
- title: News article headline
- text: Full article content
- label: Classification label (REAL/FAKE)
```

## 🛠️ Technology Stack

- **Python 3.12+**
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
  - TfidfVectorizer: Text feature extraction
  - PassiveAggressiveClassifier: Classification algorithm
  - Model evaluation metrics

## 🚀 Installation

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Purna375/AI-LearnLabs.git
   cd AI-LearnLabs
   ```

2. **Install required packages**
   ```bash
   pip install numpy pandas scikit-learn
   ```

3. **Download the dataset**
   - Ensure you have the `news.csv` file in the project directory
   - The dataset should contain columns: title, text, and label

## 📖 Usage

### Running the Jupyter Notebook

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**
   ```bash
   fake_News_Detection.ipynb
   ```

3. **Run all cells** to execute the complete pipeline

### Using the Model Programmatically

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split

# Load and prepare data
df = pd.read_csv('news.csv')
X = df['text']
y = df['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7
)

# Initialize and fit TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Train the classifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Make predictions
y_pred = pac.predict(tfidf_test)
```

## 📈 Model Performance

### Accuracy Metrics
- **Overall Accuracy**: 92.82%
- **Training Method**: 80/20 train-test split
- **Cross-validation**: Random state = 7 for reproducibility

### Confusion Matrix Results
```
                Predicted
Actual          FAKE    REAL
FAKE            591     47
REAL            44      585
```

### Performance Breakdown
- **True Positives (Real as Real)**: 585
- **True Negatives (Fake as Fake)**: 591
- **False Positives (Fake as Real)**: 47
- **False Negatives (Real as Fake)**: 44

## 🔍 How It Works

### 1. **Text Preprocessing**
- Removes English stop words (common words like "the", "and", "is")
- Applies maximum document frequency threshold (0.7)
- Converts text to lowercase and handles special characters

### 2. **Feature Extraction (TF-IDF)**
- **Term Frequency (TF)**: Measures how often a word appears in a document
- **Inverse Document Frequency (IDF)**: Measures how significant a term is across all documents
- Creates a matrix of TF-IDF features for machine learning

### 3. **Classification Algorithm**
- **Passive Aggressive Classifier**: Online learning algorithm
- Remains "passive" for correct classifications
- Becomes "aggressive" for misclassifications, updating weights
- Ideal for large-scale text classification

### 4. **Model Evaluation**
- Uses accuracy score for overall performance
- Generates confusion matrix for detailed analysis
- Provides insights into false positives and negatives

## 📁 Project Structure

```
AI-LearnLabs/
├── fake_News_Detection.ipynb    # Main Jupyter notebook
├── news.csv                     # Dataset file
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── results/                     # Output files and visualizations
    ├── confusion_matrix.png
    └── model_performance.txt
```

## 🎯 Results and Insights

### Key Findings
1. **High Accuracy**: The model achieves 92.82% accuracy, indicating strong performance
2. **Balanced Performance**: Similar precision for both REAL and FAKE news detection
3. **Low False Positive Rate**: Only 47 fake articles misclassified as real
4. **Robust Feature Extraction**: TF-IDF effectively captures important text patterns

### Model Strengths
- Fast training and prediction times
- Handles large vocabularies efficiently
- Resistant to overfitting with online learning
- Good generalization to unseen data

### Potential Improvements
- Feature engineering with n-grams
- Ensemble methods combining multiple algorithms
- Deep learning approaches (LSTM, BERT)
- Additional text preprocessing techniques

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/improvement-name
   ```
3. **Make your changes**
   - Add new features or improvements
   - Update documentation
   - Add tests for new functionality
4. **Commit your changes**
   ```bash
   git commit -m "Add: description of changes"
   ```
5. **Push to the branch**
   ```bash
   git push origin feature/improvement-name
   ```
6. **Open a Pull Request**

### Areas for Contribution
- Data preprocessing enhancements
- Additional machine learning algorithms
- Web interface for real-time detection
- API development for integration
- Performance optimization
- Documentation improvements

## 📋 Requirements

Create a `requirements.txt` file:
```
numpy>=1.26.4
pandas>=2.2.1
scikit-learn>=1.3.0
jupyter>=1.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## 🚀 Future Enhancements

- [ ] **Web Application**: Create a Flask/Django web interface
- [ ] **API Development**: REST API for real-time news classification
- [ ] **Deep Learning**: Implement BERT or other transformer models
- [ ] **Multilingual Support**: Extend to detect fake news in multiple languages
- [ ] **Real-time Processing**: Stream processing for live news feeds
- [ ] **Explainable AI**: Add model interpretability features
- [ ] **Mobile App**: Develop mobile application for on-the-go detection

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Purna375**
- GitHub: [@Purna375](https://github.com/Purna375)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/purnachandrashekar/)

## 🙏 Acknowledgments

- Dataset providers for the news classification data
- Scikit-learn community for excellent machine learning tools
- Python community for robust data science libraries
- Open source contributors who made this project possible

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/Purna375/AI-LearnLabs/issues) page
2. Create a new issue with detailed information
3. Include error messages and system information

## 📚 References

- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- [Passive Aggressive Algorithms](https://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf)
- [Text Classification with Scikit-learn](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

---

⭐ **If you found this project helpful, please give it a star!**

[![GitHub stars](https://img.shields.io/github/stars/Purna375/AI-LearnLabs.svg?style=social&label=Star)](https://github.com/Purna375/AI-LearnLabs)
[![GitHub forks](https://img.shields.io/github/forks/Purna375/AI-LearnLabs.svg?style=social&label=Fork)](https://github.com/Purna375/AI-LearnLabs/fork)

**Keywords**: `fake-news-detection` `machine-learning` `nlp` `python` `scikit-learn` `text-classification` `tfidf` `passive-aggressive-classifier`
