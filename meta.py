import praw
import re
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pandas as pd

reddit = praw.Reddit(
    client_id='CPq03s0dJxWUa54bYUB-2w',
    client_secret='sZDfM4VhJgQb5vLXebnIJjy0j3Nd9w',
    user_agent='stock-predictor by u/mirnaknez'
)

stock_keywords = {
    "Tesla": ["TSLA", "Tesla"],
    "Apple": ["AAPL", "Apple"],
    "Microsoft": ["MSFT", "Microsoft"],
    "NVIDIA": ["NVDA", "NVIDIA"],
    "Meta": ["META", "Meta", "Facebook"],
    "GameStop": ["GME", "GameStop"]
}

def collectcomments(subredditname, stocknames):
    subreddit = reddit.subreddit(subredditname)
    comments = []
    timestamps = []
    sixmonthsago = datetime.utcnow() - timedelta(days=6 * 30)
    for stockname in stocknames:
        posts = subreddit.search(stockname, limit=50)
        for post in posts:
            post.comments.replace_more(limit=0)
            for comment in post.comments.list():
                commenttime = datetime.utcfromtimestamp(comment.created_utc)
                if commenttime >= sixmonthsago:
                    comments.append(comment.body)
                    timestamps.append(commenttime)
    return comments, timestamps

def cleantext(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

def getsentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def categorizesentiment(sentimentscores):
    return [
        0 if score < -0.5 else
        1 if score < 0 else
        2 if score == 0 else
        3 if score <= 0.5 else
        4
        for score in sentimentscores
    ]

def analyzecomments(comments, timestamps, vectorizer=None):
    cleancomments = [cleantext(comment) for comment in comments]
    sentimentscores = [getsentiment(comment) for comment in comments]
    sentimentcategories = categorizesentiment(sentimentscores)
    if vectorizer is None:
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(cleancomments)
    else:
        X = vectorizer.transform(cleancomments)
    return X, sentimentcategories, sentimentscores, timestamps, vectorizer

def calculatestatistics(sentimentscores):
    q1 = np.percentile(sentimentscores, 25)
    q3 = np.percentile(sentimentscores, 75)
    mean = np.mean(sentimentscores)
    median = np.median(sentimentscores)
    stddev = np.std(sentimentscores)
    variance = np.var(sentimentscores)
    mode = pd.Series(sentimentscores).mode().iloc[0]
    return {
        "Mean": mean,
        "Median": median,
        "Q1": q1,
        "Q3": q3,
        "Standard Deviation": stddev,
        "Variance": variance,
        "Mode": mode
    }

def scaledata(data, targetmin, targetmax):
    scaler = MinMaxScaler(feature_range=(targetmin, targetmax))
    return scaler.fit_transform(np.array(data).reshape(-1, 1)).flatten()

def getstockdata(ticker, startdate, enddate):
    stock = yf.Ticker(ticker)
    data = stock.history(start=startdate, end=enddate)
    return data

def dailysentiment(sentimenttimes, sentimentscores):
    df = pd.DataFrame({'Time': sentimenttimes, 'Sentiment': sentimentscores})
    df.set_index('Time', inplace=True)
    dailyaverage = df.resample('D').mean()
    dailyaverage = dailyaverage.interpolate(method='linear')
    return dailyaverage.index, dailyaverage['Sentiment']

def weekpredict(sentimenttimes, sentimentscores):
    df = pd.DataFrame({'Time': sentimenttimes, 'Sentiment': sentimentscores})
    df.set_index('Time', inplace=True)
    lastweek = df[df.index >= (df.index.max() - pd.Timedelta(days=7))]
    totalpolarity = lastweek['Sentiment'].mean()
    if totalpolarity > 0:
        return "Prediction based on comments from the past week: META stock price will rise."
    elif totalpolarity < 0:
        return "Prediction based on comments from the past week: META stock price will fall."
    else:
        return "Prediction based on comments from the past week: META stock price will not rise or fall."

def comparegraph(sentimenttimes, sentimentscores, stockdata):
    dailytimes, dailysentiments = dailysentiment(sentimenttimes, sentimentscores)
    stockdatafiltered = stockdata[stockdata.index >= (stockdata.index.max() - pd.DateOffset(months=6))]
    smoothedsentiments = scaledata(dailysentiments, stockdatafiltered['Close'].min(), stockdatafiltered['Close'].max())
    plt.figure(figsize=(12, 6))
    plt.plot(dailytimes, smoothedsentiments, label="Average sentiment", color="blue", alpha=0.7)
    plt.plot(stockdatafiltered.index, stockdatafiltered['Close'], label="Stock price", color="green")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Comparison of average sentiment and META stock price")
    plt.legend()
    plt.grid()
    plt.show()
    prediction = weekpredict(dailytimes, dailysentiments)
    print(prediction)

def lastweekdata(comments, timestamps, vectorizer):
    df = pd.DataFrame({'Comment': comments, 'Time': pd.to_datetime(timestamps)})
    df.set_index('Time', inplace=True)
    lastweek = df[df.index >= (df.index.max() - pd.Timedelta(days=7))]
    return vectorizer.transform(lastweek['Comment']), lastweek.index

def boxplot(sentimentscores):
    plt.figure(figsize=(10, 6))
    plt.boxplot(sentimentscores, vert=True, patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="blue"),
                medianprops=dict(color="red", linewidth=2),
                whiskerprops=dict(color="blue", linestyle="-"),
                capprops=dict(color="blue"),
                flierprops=dict(marker="o", color="red", alpha=0.5))
    plt.title("Meta", fontsize=16)
    plt.ylabel("Sentiment scores", fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
    
def main():
    comments1, timestamps1 = collectcomments("wallstreetbets", stock_keywords["Meta"])
    comments2, timestamps2 = collectcomments("wallstreetbetsnew", stock_keywords["Meta"])
    comments = comments1 + comments2
    timestamps = timestamps1 + timestamps2
    X, sentimentcategories, sentimentscores, timestamps, vectorizer = analyzecomments(comments, timestamps)

    stats = calculatestatistics(sentimentscores)
    print("Sentiment Statistics:")
    for stat, value in stats.items():
        print(f"{stat}: {value}")
    boxplot(sentimentscores)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, sentimentcategories, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    print("Classification Results:")
    print(classification_report(ytest, ypred, target_names=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]))

    sentimentdistribution = pd.Series(sentimentcategories).value_counts()
    print("Sentiment Distribution:")
    print(sentimentdistribution)

    stockdata = getstockdata("META", min(timestamps).strftime("%Y-%m-%d"), max(timestamps).strftime("%Y-%m-%d"))
    comparegraph(timestamps, sentimentscores, stockdata)
    
    lastweekX, _ = lastweekdata(comments, timestamps, vectorizer)
    yweek = model.predict(lastweekX)
    majorityclass = np.mean(yweek)
    if majorityclass > 2.0:
        print("Model prediction for the past week: META stock price will rise.")
    elif majorityclass < 2.0:
        print("Model prediction for the past week: META stock price will fall.")
    else:
        print("Model prediction for the past week: META stock price will not rise or fall.")

if __name__ == "__main__":
    main()
