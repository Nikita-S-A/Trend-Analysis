
# 🏡 Airbnb-Themed Property Topic Modeling using NLP

## 📘 Overview

This project leverages Natural Language Processing (NLP) techniques to identify trending themes in Airbnb guest messages to guide the design of themed rental properties. The goal is to provide Airbnb hosts with actionable insights into which pop culture references—like movies, TV shows, or unique experiences—are resonating with users, allowing for more personalized, profitable property designs.

---

## 🔍 What the Project Does

This project uses multiple NLP models—including LDA, Seeded LDA, and BERT—for topic modeling and sentiment analysis of Airbnb guest data. It identifies popular themes that users mention and ranks them based on importance, frequency, and sentiment.

---

## 💡 Why the Project Is Useful

Understanding guests’ interests allows Airbnb hosts to design properties tailored to high-demand themes (e.g., "Wes Anderson", "Godzilla", "Alien vs Predator"). This personalization enhances guest experience, improves occupancy rates, and boosts revenue.

---

## 🚀 How to Get Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/airbnb-themed-topic-modeling.git
cd airbnb-themed-topic-modeling
```

### 2. Prepare Your Environment
Install required Python libraries:
```bash
pip install -r requirements.txt
```

### 3. Place Your Input Data
Ensure your input file (`input.csv` or `wanderlust.csv`) is in the project directory. The expected column is `MsgBody`.

### 4. Run the Scripts
- **Preprocessing**:  
  ```bash
  python text_preprocessing.py
  ```

- **Topic Modeling with LDA**:  
  ```bash
  python LDA.py
  ```

- **Guided LDA with Seed Words**:  
  ```bash
  python Guided\ LDA.py
  ```

- **BERT Sentiment & Embedding**:  
  ```bash
  python Bert4.py
  ```

---

## 📂 Repository Structure

```
.
├── Bert4.py                  # BERT sentiment and embedding extraction
├── Guided LDA.py             # Seeded LDA topic modeling
├── LDA.py                    # Traditional LDA model
├── text_preprocessing.py     # Preprocessing script
├── input.csv                 # Input dataset (MsgBody column)
├── processed_text.csv        # Output from preprocessing
├── merge.csv                 # Final results with keywords, sentiment, and TF-IDF
├── airbnb-Pitch-Deck.pptx    # Final presentation deck
└── README.md                 # Project overview
```

---

## 🧰 Technologies Used

| Tool/Library    | Purpose                                  |
|------------------|-------------------------------------------|
| Python           | Programming language                     |
| Gensim           | LDA and Guided LDA modeling              |
| Transformers     | BERT & RoBERTa-based NLP                 |
| TextBlob         | Sentiment analysis                       |
| NLTK             | Tokenization, stopwords, lemmatization   |
| scikit-learn     | TF-IDF, classification utilities         |
| matplotlib/seaborn | Data visualization                     |
| LangChain + LLaMA2 | Advanced categorization (explored)     |

---

## 📈 Key Pivots in Project

1. **LDA** – Initial topic modeling with moderate success.
2. **Bigram/Trigram LDA** – Tried to improve phrase detection, but added noise.
3. **Seeded LDA** – Focused on specific topics via custom seed words.
4. **BERT/RoBERTa** – Sentiment analysis and topic clustering, high accuracy but resource-intensive.
5. **LangChain + LLaMA2** – Advanced topic grouping for text comprehension.

---

## 🔍 Key Findings

- Repeated themes include: **Godzilla & King Kong**, **Alien vs Predator**, **Wes Anderson films**
- BERT sentiment confirmed positive user affinity for **Wes Anderson-inspired themes**
- Seeded LDA was limited by dataset diversity; BERT models performed better but required more resources

---

## 👥 Project Contributors
  
- Nikita Singh 

---

## 📄 License

This project is for academic and learning use only and is not licensed for commercial applications.

---

## 📬 Contact

For collaboration or inquiries:

**Nikita Singh**  
📧 Email: [mail2nikita95@gmail.com] 
🔗 LinkedIn: [linkedin.com/in/nikitasingh3](https://linkedin.com/in/nikitasingh3)
