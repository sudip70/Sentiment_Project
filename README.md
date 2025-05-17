# 🏥 Sentiment\_Project

This project proposes a **sentiment analysis system** designed to classify **patient feedback** on healthcare services. By leveraging machine learning techniques, the system aims to understand patient sentiments (**positive or negative**) and provide actionable insights for healthcare providers.

GitHub Repo: [Sentiment\_Project](https://github.com/sudip70/Sentiment_Project)

---

## 🔧 Features

* Classifies patient feedback as **positive** or **negative**
* Pre-trained logistic regression model using healthcare-related review data
* Applies NLP preprocessing steps like **cleaning**, **tokenization**, and **lemmatization**
* Real-time text input through a **Streamlit web interface**

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sudip70/Sentiment_Project.git
cd Sentiment_Project
```

### 2. Install Dependencies

```bash
pip install streamlit pandas scikit-learn joblib
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📦 Project Structure

```
📁 Sentiment_Project/
├── app.py              # Streamlit web app
├── model.pkl           # Pre-trained logistic regression model
├── vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt    # List of dependencies
```

---

## 💡 How It Works

1. User submits patient feedback via the web interface.
2. Text is preprocessed and vectorized using a TF-IDF model.
3. Sentiment is predicted using a trained logistic regression model.
4. Sentiment result is returned as **positive** or **negative** or **neutral**.

---

## 📊 Dataset & Approach

* Uses existing sentiment analysis datasets (e.g., Twitter, Reddit, Google Reviews)
* Focused on healthcare-related reviews to simulate patient feedback
* Applies text cleaning, tokenization, and lemmatization for preprocessing
* Aims to provide measurable outcomes and support healthcare providers in enhancing patient experience

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Pull requests and issues are welcome. For major changes, please open an issue to discuss proposed updates.

---

## 🙌 Acknowledgements

* Built with Python, Scikit-learn, and Streamlit
* Inspired by NLP applications in healthcare and sentiment mining techniques
