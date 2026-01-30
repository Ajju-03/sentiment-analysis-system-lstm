# Sentiment Analysis system using LSTM 
* Built a Deep Learning based sentiment analysis system used to classify text as positive or negative.
* Built using machine learning algorithms using python
   libraries like scikit-learn,tensorflow,keras, numpy, pandas, joblib and fastapi for model serving.

# Project Structure 
📦 sentiment-analysis-system-lstm
├── data/
|   └── IMDB Dataset.csv
|
|
├── 📂 Notebook/
|      ├── tuner_logs/lstm-sentiment/
|      ├── preprocessing.py
|      ├── sentiment_analysis.ipynb
|      └── sentiment_analysis2.ipynb
|
├──  main.py  
├── sentiment_analysis2.keras
├── tokenizer.pkl
└── README.md

# Structure Info

### 📂 data/
Collected dataset from kaggle IMDB dataset 50K.

### 📂 Notebook/
Implemented logic for sentiment analysis and analyzed results.

- `tuner_logs/lstm-sentiment/` – used keras tuner to perform tuning for lstm model
- `preprocessing.py` – defined functions for preprocessing text like removing html tags, punctuation and lowering the text
- `sentiment_analysis.ipynb` – used scikit learn and text preprocessing techniques for base model using Logistic Regression and TF-IDF for vectorization.
- `sentiment_analysis2.ipynb` – Used deep learning based lstm model for training & evaluated and achieved 89% of test accuracy.
  
**main.py**
Served model using FastAPI
**sentiment_analysis2.keras**
Saved the lstm model using tensorflow
**tokenizer.pkl**
saved the tokenizer model using joblib

## ⚙️ How It Works

1. The user enters reviews via JSON endpoints in swagger UI using fastapi .

2. The submitted data is sent to the FastAPI backend via an HTTP POST request.

3. The backend preprocesses the input data by:
   - Converting it into a tokens and analyzes the given data.

4. The preprocessed data is passed to the trained machine learning model to generate:
   - A positive or negative to classify text
   - A probability score indicating possibility for sentiment


# Tech Stack ⚙️

| Technology | Usage |
|-----------|--------|
| Python | Backend logic |
| Scikit-learn | Machine learning |
| TensorFlow | Deep Learning |
| keras | API |
| NLP | Core NLP |
| FastAPI | API framework |
| Pandas | Data processing |
| NumPy | Numerical computation ||


## 🚀 Run Locally

### 1. Clone the Repository
```bash
(https://github.com/Ajju-03/sentiment-analysis-system-lstm.git) 
### 2. Create a Virtual Environment

python -m venv venv

### 3. Activate it

**Windows**

venv\Scripts\activate

**MacOS**

source venv/bin/activate

### 3. install dependencies

pip install -r requirements.txt

### 4. Start the Application
uvicorn app.main:app --reload

### API Documentation
FastAPI automatically generates interactive API docs:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
