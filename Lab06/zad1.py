import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

lem = WordNetLemmatizer()
stem = PorterStemmer()
text_file = open("article.txt", "r")
text = text_file.read()
text_file.close()
tokenized_text=sent_tokenize(text)
tokenized_word = word_tokenize(text)
fdist = FreqDist(tokenized_word)
print(fdist)
fdist.plot(30, cumulative=False)
plt.savefig('plot.png')
plt.show()
new_stopwords=["the",".","'","to","of","that","and","'s","has","is"]
stopwords=nltk.corpus.stopwords.words('english')
stopwords.extend(new_stopwords)
filtered_sent=[]
for w in tokenized_word:
    if w not in stopwords:
        filtered_sent.append(w)

stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(stem.stem(lem.lemmatize(w,"v")))
# print("Stemmed Sentence:",stemmed_words)
filtered=[]
for w in stemmed_words:
    if not w in stopwords:
        filtered.append(w)

print len(text)
print len(tokenized_text)
print len(tokenized_word)
print len(stemmed_words)
print len(filtered)
print filtered
nltk.pos_tag(filtered)

ammount_of_W = FreqDist(filtered)
ammount_of_W.plot(30,cumulative=False)
print(ammount_of_W)
