import streamlit as st
import re
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from textblob import TextBlob
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim import corpora
import string
from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.style import set_palette
from gensim.parsing.preprocessing import remove_stopwords
import snscrape.modules.twitter as sntwitter
import nltk
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')

plt.rcParams['figure.figsize'] = (20.0, 20.0)
plt.rc('font', size=16)
set_palette('flatui')
st.markdown("<meta name='image' property='og:image' content='cool.jpg'>", unsafe_allow_html=True)

# Location = 'London, United Kingdom'
# Distance = '200mi'

#App Start
def app():
    st.title("Stock Tweet Analyzer 📈")

    st.subheader("Analyze the tweets of your favorite stocks")
    
    st.subheader("Watch me first!")
    video_file = open("apiUpdateVideo.webm","rb")
    video_bytes = video_file.read()
    st.video(video_bytes)

    # User Input Section area
    raw_text_U = st.text_area("What stock are we looking up today? - eg. tesla, facebook, apple")
    count = st.slider("How many tweet's would you like to pull", min_value=500, max_value=3000, step=100)
    try:
        Start_date = st.date_input("Date range of tweets", [])

        since_date = str(Start_date[0])
        until_date = str(Start_date[1])
    except (IndexError, NameError):
        pass

    # raw_text = '$' + raw_text_U # <-- Search Stock Based on Ticker
    raw_text = raw_text_U + ' stock'
    raw_text = raw_text.lower()
    # raw_text = raw_text_U + ' stock'

    # Choice Box selection
    Analyzer_choice = st.selectbox("What would you like to find out?",
                                   ["Stock Sentiment", "WordCloud Generation",
                                    "Top 10 Words associated with the stock", "Stock Theme",
                                    "Charts & Graphs of buyer positions"])
    try:
        try:
            if st.button("Analyze"): # Button

                # Sentiment Area Start
                if Analyzer_choice == "Stock Sentiment":
                    st.subheader(
                        'This stock sentiment Analyzer uses NLP based technology to interpret and classify emotions '
                        'of text data ')
                    my_placeholder = st.empty()
                    message = "Fetching Tweets"
                    st.success(message)

                    # Function for getting tweets

                    def Show_Recent_Tweets(raw_text):
                    #     # Collecting tweets for Dataframe
                    
                        tweets_list2 = []

                        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(raw_text + ' since:' + since_date +' until:' + until_date).get_items()):
                            if i>count:
                                break
                            tweets_list2.append([tweet.content])
                        
                        df = pd.DataFrame(tweets_list2, columns=['Tweet'])

                        # Cleaning Tweets
                        df['cleanLinks'] = df['Tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])  # Removing URLs
                        df['cleanLinks'] = df['cleanLinks'].apply(lambda x: x.lower())  # applying lowercase to text

                        # Special Character list
                        spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
                                      "*", "+", ",", "-", ".", "/", ":", ";", "<",
                                      "=", ">", "?", "@", "[", "\\", "]", "^", "_",
                                      "`", "{", "|", "}", "~", "–", '$']

                        for char in spec_chars:
                            df['cleanLinks'] = df['cleanLinks'].str.replace(char, ' ')

                            # Applying stock market sentiment
                        df['Sentiment'] = df['cleanLinks'].apply(lambda x: TextBlob(x))
                        df['Sentiment'] = df['Sentiment'].apply(lambda x: x.sentiment)
                        df['Polarity'] = df['Sentiment'].apply(lambda x: re.split(',', str(x))[0])
                        df['Polarity'] = df['Polarity'].apply(lambda x: re.split('=', str(x))[1])
                        df['Polarity'] = df['Polarity'].apply(lambda x: float(x))

                        df['Subjectivity'] = df['Sentiment'].apply(lambda x: re.split(',', str(x))[1])
                        df['Subjectivity'] = df['Subjectivity'].apply(lambda x: re.split('=', str(x))[1])
                        df['Subjectivity'] = df['Subjectivity'].apply(lambda x: x.strip(')'))
                        df['Subjectivity'] = df['Subjectivity'].apply(lambda x: float(x))

                        # Determine the subjectivity we are using * Will PLay round with this number
                        df = df[df.Subjectivity > 0.5]

                        # Polarity score
                        score = round(df['Polarity'].mean(), ndigits=3)

                        # Display Text

                        sample_tweet = df['cleanLinks'].iloc[0]
                        st.markdown('**Sample Tweet: **' + sample_tweet)

                        st.write(
                            'The general sentiment for ' + raw_text_U + "'s stock was " + str(
                                score) + ' On a scale of 1 to -1')
                        if score > 0:
                            st.markdown('This reflects a **positive** sentiment')
                        else:
                            st.markdown('This reflects a **negative** sentiment')

                        st.markdown('**Graphs of sentiment over time:**')

                        st.area_chart(df['Polarity'])

                        return df

                    try:
                        Show_Recent_Tweets(raw_text)
                    except IndexError:
                        st.error('**Not enough tweets found to generate sentiment**')
                    st.success(my_placeholder)

                # Wordcloud generation
                elif Analyzer_choice == "WordCloud Generation":
                    st.subheader(' A visual representation of ' + raw_text_U + "'s tweets")

                    message = 'Generating WordCloud'

                    st.success(message)

                    # Create Word Cloud
                    def gen_wordcloud():

                        tweets_list2 = []

                        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(raw_text + ' since:' + since_date +' until:' + until_date).get_items()):
                            if i>count:
                                break
                            tweets_list2.append([tweet.content])
                        
                        df = pd.DataFrame(tweets_list2, columns=['Tweet'])

                        df['cleanLinks'] = df['Tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])  # Removing URLs
                        df['cleanLinks'] = df['cleanLinks'].apply(lambda x: x.lower())  # applying lowercase to text

                        # Special Character list
                        spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
                                      "*", "+", ",", "-", ".", "/", ":", ";", "<",
                                      "=", ">", "?", "@", "[", "\\", "]", "^", "_",
                                      "`", "{", "|", "}", "~", "–", '$']

                        for char in spec_chars:
                            df['cleanLinks'] = df['cleanLinks'].str.replace(char, ' ')
                        # WC generation
                        words = " ".join(df['cleanLinks'])

                        # remove punctuation and stop words
                        def punctuation_stop(text):

                            filtered = []
                            stop_words = set(stopwords.words('english'))
                            word_tokens = word_tokenize(text)
                            for w in word_tokens:
                                if w not in stop_words and w.isalpha():
                                    filtered.append(w.lower())
                            return filtered

                        unwanted = [raw_text, raw_text_U, 'market', 'moving', 'average', 'economy', 'stockmarket',
                                    'stocks', 'stock', 'people', 'money', 'markets', 'today', 'http', 'the', 'to', 'and',
                                    'is',
                                    'of',
                                    'in', 'it', 'you', 'for', 'on', 'this', 'will', 'are', 'price', 'dow', 'jones',
                                    'robinhood', 'link', 'http', 'dow', 'jones', 'order', '//', 'sign', 'join', 'claim']
                        try:
                            words_filtered = punctuation_stop(words)
                            text = " ".join([ele for ele in words_filtered if ele not in unwanted])
                            wc = WordCloud(background_color="gray", stopwords=STOPWORDS, max_words=500, width=2000,
                                           height=2000)
                            wc.generate(text)
                            plt.imshow(wc, interpolation="bilinear")
                            plt.axis('off')
                            plt.savefig('WC.png')
                            gen = Image.open("WC.png")
                            plt.show()
                            return gen

                        except ValueError:
                            st.error('**Not enough tweets found to build wordcloud**')

                    try:
                        gen = gen_wordcloud()

                        st.image(gen, use_column_width=True)
                    except AttributeError:
                        pass


                # Buyers Charts and graphs
                elif Analyzer_choice == "Charts & Graphs of buyer positions":

                    st.success("Constructing Graphs")

                    def graphs():

                        tweets_list2 = []

                        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(raw_text + ' since:' + since_date +' until:' + until_date).get_items()):
                            if i>count:
                                break
                            tweets_list2.append([tweet.content])
                        
                        df = pd.DataFrame(tweets_list2, columns=['Tweet'])

                        # Create Dataframe with just tweets
                        df['cleanLinks'] = df['Tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])  # Removing URLs
                        df['cleanLinks'] = df['cleanLinks'].apply(lambda x: x.lower())  # applying lowercase to text

                        # Special Characters list
                        spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
                                      "*", "+", ",", "-", ".", "/", ":", ";", "<",
                                      "=", ">", "?", "@", "[", "\\", "]", "^", "_",
                                      "`", "{", "|", "}", "~", "–", '$']

                        for char in spec_chars:
                            df['cleanLinks'] = df['cleanLinks'].str.replace(char, ' ')
                            # Counting Numbers

                        def Wordcount(cleanLinks):
                            if 'buying' in cleanLinks.lower():
                                return 'buy positions mentioned'
                            if 'selling' in cleanLinks.lower():
                                return 'sell positions mentioned'
                            if 'buy' in cleanLinks.lower():
                                return 'buy positions mentioned'
                            if 'sell' in cleanLinks.lower():
                                return 'sell positions mentioned'
                            if 'short' in cleanLinks.lower():
                                return 'short positions mentioned'
                            if 'long' in cleanLinks.lower():
                                return 'long positions mentioned'
                            if 'put' in cleanLinks.lower():
                                return 'puts mentioned'
                            if 'call' in cleanLinks.lower():
                                return 'calls mentioned'
                            else:
                                return

                        df['Market Polar Position'] = df['cleanLinks'].apply(Wordcount)

                        # Graph numbers
                        st.markdown('**Visualization of investor positions:**')
                        position_A = df['Market Polar Position'].value_counts()
                        st.write(position_A)

                        # Graph
                        plt.axis('off')
                        df['Market Polar Position'].value_counts().plot(kind='pie', autopct='%1.1f%%',
                                                                        figsize=(10, 5))
                        plt.savefig('buyers.png')
                        buy = Image.open("buyers.png")
                        return buy

                    buy = graphs()

                    st.image(buy)

                # Gensim Stock Theme
                elif Analyzer_choice == "Stock Theme":
                    st.subheader('the following represents an AI Generated topic model for ' + raw_text_U + "'s stock")
                    st.success("Constructing theme from tweets")
                    
                    tweets_list2 = []

                    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(raw_text + ' since:' + since_date +' until:' + until_date).get_items()):
                        if i>count:
                            break
                        tweets_list2.append([tweet.content])
                    
                    df = pd.DataFrame(tweets_list2, columns=['Tweet'])

                    # Create Dataframe with just tweets
                    df['cleanLinks'] = df['Tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])  # Removing URLs
                    df['cleanLinks'] = df['cleanLinks'].apply(lambda x: x.lower())  # applying lowercase to text

                    # Special Character list
                    spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
                                  "*", "+", ",", "-", ".", "/", ":", ";", "<",
                                  "=", ">", "?", "@", "[", "\\", "]", "^", "_",
                                  "`", "{", "|", "}", "~", "–", '$']

                    for char in spec_chars:
                        df['cleanLinks'] = df['cleanLinks'].str.replace(char, ' ')

                    # preprocessing sentence structure
                    stop = set(stopwords.words('english'))
                    exclude = set(string.punctuation)
                    lemma = WordNetLemmatizer()

                    def clean(doc):
                        stop_free = " ".join([word for word in doc.lower().split() if word not in stop])
                        punc_free = "".join(token for token in stop_free if token not in exclude)
                        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
                        return normalized

                    doc_clean = [clean(comment).split() for comment in df['cleanLinks']]

                    dictionary = corpora.Dictionary(doc_clean)
                    corpus = [dictionary.doc2bow(text) for text in doc_clean]

                    # LDA Model
                    try:
                        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

                        x = ldamodel.show_topics()

                        twords = {}
                        for topic, word in x:
                            twords[topic] = re.sub('[^A-Za-z ]+', '', word)
                            # Create 3 topics
                        st.write('Topic 1 generated: ' + twords[0])
                        st.write('Topic 2 generated: ' + twords[1])
                        st.write('Topic 3 generated: ' + twords[2])
                    except ValueError:
                        st.error('**Not enough tweets found to generate word count**')
                # Top word count theme
                else:
                    # Top Words mentioned from the stock
                    def top_words():
                        st.subheader("The top words mentioned in " + raw_text_U + " tweets:")
                        st.success("Processing Word Count")
                        
                        tweets_list2 = []
                        
                        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(raw_text + ' since:' + since_date +' until:' + until_date).get_items()):
                            if i>count:
                                break
                            tweets_list2.append([tweet.content])
                        
                        df = pd.DataFrame(tweets_list2, columns=['Tweet'])

                        # Create Dataframe with just tweets
                        df['Tweets'] = df['Tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])  # Removing URLs
                        df['Tweets'] = df['Tweets'].apply(lambda x: x.lower())  # applying lowercase to text

                        # Special Character list
                        spec_chars1 = ["!", '"', "#", "%", "&", "'", "(", ")",
                                       "*", "+", ",", "-", ".", "/", ":", ";", "<",
                                       "=", ">", "?", "@", "[", "\\", "]", "^", "_",
                                       "`", "{", "|", "}", "~", "–", '$']

                        for char1 in spec_chars1:
                            df['Tweets'] = df['Tweets'].str.replace(char1, ' ')
                            text = df.Tweets.str.cat(sep=' ')
                            filtered_sentence = remove_stopwords(text)

                        # unwanted word list (Experimental)
                        unwanted = [raw_text, raw_text_U, 'market', 'moving', 'average', 'economy', 'stockmarket',
                                    'stocks', 'stock', 'people', 'money', 'markets', 'today', 'http', 'the', 'to', 'and',
                                    'is',
                                    'of',
                                    'in', 'it', 'you', 'for', 'on', 'this', 'will', 'are', 'price', 'dow', 'jones']

                        # vectorized text

                        def get_top_n_words(corpus, n=None):
                            vec = CountVectorizer().fit(corpus)
                            bag_of_words = vec.transform(corpus)
                            sum_words = bag_of_words.sum(axis=0)
                            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
                            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
                            return words_freq[:n]

                        text = df.Tweets.str.cat(sep=' ')
                        filtered_sentence = remove_stopwords(text)
                        texa = {'texz': [filtered_sentence]}
                        dfz = pd.DataFrame(data=texa)

                        common_words = get_top_n_words(dfz['texz'], 20)

                        plt.figure(figsize=(18, 8))
                        plt.rc('xtick', labelsize=15)

                        df1 = pd.DataFrame(common_words, columns=['Tweets', 'count'])
                        # plt.xaxis('off')
                        df1.groupby('Tweets').sum()['count'].sort_values(ascending=False).plot(
                            kind='bar', rot=30, fontsize=12)
                        plt.savefig('top_10.JPEG')
                        top = Image.open("top_10.JPEG")
                        plt.show()
                        return top

                    try:
                        top = top_words()

                        st.image(top, use_column_width=True, width=100000,
                                 caption='The Top 20 most frequent words in ' + raw_text + ' tweets\n')
                    except ValueError:
                        st.error('**Not enough tweets found to generate word count**')

            # Credits
            st.subheader('Author: Samuel Lawrence ')
            st.write(
                'Disclaimer: This content is intended to be used and must be used for informational purposes only. It is '
                'very important to do your own analysis before making any investment based on your own personal '
                'circumstances.')
            st.write('Github repo: https://github.com/SamuelLawrence876/Twitter-Sentiment-Stocks-deployment')

            # Error Exceptions
        except NameError:
            st.error("Please enter a range")
    except SystemExit:
        st.error("Too many requests, please try again later")



if __name__ == "__main__":
    app()

# Fix graph for sentiment & round sentiment figure
