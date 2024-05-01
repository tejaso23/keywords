import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist

nltk.download('punkt')
nltk.download('stopwords')


def summarize_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Stemming the words
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in filtered_words]

    # Calculate frequency distribution of words
    freq_dist = FreqDist(stemmed_words)

    # Assign score to each sentence based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq_dist.keys():
                if len(sentence.split(' ')) < 30:
                    if sentence not in sentence_scores.keys():
                        sentence_scores[sentence] = freq_dist[word]
                    else:
                        sentence_scores[sentence] += freq_dist[word]

    # Get the top 3 sentences with highest scores
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]

    return ' '.join(summarized_sentences)


# Example usage:
text = "Your long piece of text goes here."
summary = summarize_text(text)
print(summary)
