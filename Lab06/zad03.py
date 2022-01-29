import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentences = ["They picked me up from the station. The hotel is surrounded by nature, and I enjoyed exploring around (The small path in front of the hotel takes you to a small forest? and is easy to walk through with ordinary shoes, but the backyard is a bit tricky - Basically, it’s full of wild grasses that could irritate your skin. I guess it would be lovelier in winter with snowshoes! I also liked the footbridge between cottages and the main hotel site). My room was just above the reception. It was clean and spacious. The vegetarian breakfast was delicious. It’s challenging to find a hotel that serves vegetarian food in Japan, so I’m happy about that. I had a very refreshing weekend. I’d love to come back with my partner (when Japan opens up the border!) ",
"Despite being only a year old, the building was full of cracks, had a bug, and the floor on the first floor was not put together properly.  The shower head was incredibly hard to use, the view of the mountain was obstructed by a parking lot, hotel, and road, glasses were dirty, and previous guests garbage and ashtrays were left outside."]

for sentence in sentences:
        sid = SentimentIntensityAnalyzer()
        print(sentence)
        ss = sid.polarity_scores(sentence)
        for k in sorted(ss):
            print('{0}: {1}, '.format(k, ss[k]), end='')
        print()
