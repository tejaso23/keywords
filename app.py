import pickle
from flask import Flask, render_template, request
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from heapq import nlargest
from collections import Counter

app = Flask(__name__)

# Load pickled files & data
with open('count_vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

with open('tfidf_transformer.pkl', 'rb') as f:
    tfidf_transformer = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Cleaning data:
stop_words = set(stopwords.words('english'))
new_stop_words = ["fig","figure","image","sample","using",
             "show", "result", "large",
             "also", "one", "two", "three",
             "four", "five", "seven","eight","nine"]
stop_words = list(stop_words.union(new_stop_words))

def preprocess_text(txt):
    # Lower case
    txt = txt.lower()
    # Remove HTML tags
    txt = re.sub(r"<.*?>", " ", txt)
    # Remove special characters and digits
    txt = re.sub(r"[^a-zA-Z]", " ", txt)
    # tokenization
    txt = nltk.word_tokenize(txt)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    txt = [word for word in txt if word not in stop_words]
    # Remove words less than three letters
    txt = [word for word in txt if len(word) >= 3]
    # Lemmatize
    lmtr = WordNetLemmatizer()
    txt = [lmtr.lemmatize(word) for word in txt]

    return " ".join(txt)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results

def summarize_text(text):
    """Summarizes the uploaded text using sentence scoring."""
    sentences = nltk.sent_tokenize(text)
    sentence_scores = [len(sentence.split()) for sentence in sentences]
    sorted_idx = sorted(range(len(sentence_scores)), key=sentence_scores.__getitem__, reverse=True)[:5]  # Top 5 sentences
    summary = " ".join([sentences[i] for i in sorted_idx])
    return summary

def summarization_method1(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    # Calculate the frequency of each word
    word_freq = Counter(words)
    # Get the most common words (you can adjust the number as needed)
    most_common_words = word_freq.most_common(10)
    # Select sentences containing these common words
    summary_sentences = []
    for sentence in nltk.sent_tokenize(text):
        if any(word in nltk.word_tokenize(sentence) for word, _ in most_common_words):
            summary_sentences.append(sentence)
    # Join the selected sentences to form the summary
    summary = ' '.join(summary_sentences)
    return summary

def summarization_method2(text):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    # Calculate the length of each sentence
    sentence_lengths = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
    # Calculate the average sentence length
    average_length = sum(sentence_lengths) / len(sentence_lengths)
    # Select sentences that are above the average length
    selected_sentences = [sentence for sentence in sentences if len(nltk.word_tokenize(sentence)) > average_length]
    # Calculate the scores for the selected sentences (simple method: length-based scoring)
    sentence_scores = {sentence: len(nltk.word_tokenize(sentence)) for sentence in selected_sentences}
    # Select the top 5 sentences based on scores
    summary_sentences = nlargest(5, sentence_scores, key=sentence_scores.get)
    # Join the selected sentences to form the summary
    summary = ' '.join(summary_sentences)
    return summary



@app.route('/')
def index():
    return render_template('index.html')







@app.route('/search_keywords', methods=['POST'])
def search_keywords():
    search_query = request.form['search']
    if search_query:
        keywords = []
        for keyword in feature_names:
            if search_query.lower() in keyword.lower():
                keywords.append(keyword)
                if len(keywords) == 20:  # Limit to 20 keywords
                    break
        return render_template('keywordslist.html', keywords=keywords)
    return render_template('index.html')


from langdetect import detect
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize NLTK resources (download required resources)
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return 'en'  # Default to English if language detection fails

def analyze_sentiment(text, lang):
    if lang == 'en':
        # Analyze sentiment for English text
        scores = sid.polarity_scores(text)
        compound_score = scores['compound']
        if compound_score >= 0.05:
            sentiment = 'Positive'
        elif compound_score <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
    else:
        sentiment = 'Language not supported'
    return sentiment


@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    document = request.files['file']
    if document.filename == '':
        return render_template('index.html', error='No document selected')

    if document:
        text = document.read().decode('utf-8', errors='ignore')
        summary = summarize_text(text)
        lang = detect_language(text)
        sentiment = analyze_sentiment(text, lang)
        summary_method1 = summarization_method1(text)
        summary_method2 = summarization_method2(text)
        preprocessed_text = preprocess_text(text)
        tf_idf_vector = tfidf_transformer.transform(cv.transform([preprocessed_text]))
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        keywords = extract_topn_from_vector(feature_names, sorted_items, 20)
        keywords_list = list(keywords.items())
        return render_template('keywords.html', summary=summary, summary1=summary_method1, summary2=summary_method2,
                               keywords=keywords_list, sentiment=sentiment)  # Pass sentiment to template
        return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)