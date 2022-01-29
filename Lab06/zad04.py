import snscrape.modules.twitter as sntwitter
import pandas
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


tweet_list = []

for i,tweet in enumerate(sntwitter.TwitterSearchScraper('hashtag trump since:2021-11-01 until:2022-01-21').get_items()):
    if i>500:
        break
    tweet_list.append([tweet.content])

listToStr = ' '.join(map(str, tweet_list))

lem = WordNetLemmatizer()
stem = PorterStemmer()

tokenized_text=sent_tokenize(listToStr)
tokenized_word = word_tokenize(listToStr)
fdist = FreqDist(tokenized_word)
# print(fdist)
# fdist.plot(30,cumulative=False)

plt.show()
new_stopwords=[",",".","'s","mr",';',"ms","=","''",'....',"``","-",'’',".","[","]","#","hashtag","@", ",","a",":","is","?",]
stopwrd=nltk.corpus.stopwords.words('english')
stopwrd.extend(new_stopwords)
filtered_sent=[]
for w in tokenized_word:
    if not w in stopwrd:
        filtered_sent.append(w)

stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(stem.stem(lem.lemmatize(w,"v")))

filtered=[word for word in stemmed_words if not word in stopwrd]

print filtered

ammount_of_W = FreqDist(filtered)
ammount_of_W.plot(30,cumulative=False)
print ammount_of_W
