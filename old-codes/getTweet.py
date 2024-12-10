import snscrape.modules.twitter as sntwitter


# Linkten tweet ID'sini al
def extract_tweet_id(url):
    return url.split("/")[-1]


# Belirli bir tweetin altındaki yorumları çek
def get_replies(tweet_url):
    tweet_id = extract_tweet_id(tweet_url)
    query = f"conversation_id:{tweet_id}"  # Tweet'e ait konuşmayı arar
    replies = []

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if tweet.inReplyToTweetId == int(tweet_id):  # Belirli tweet'e cevap olanları filtrele
            replies.append(tweet.content)

    return replies


# Örnek Kullanım
tweet_url = "https://twitter.com/pmcafrica/status/1866129845698093402"
comments = get_replies(tweet_url)

# Yorumları yazdır
for i, comment in enumerate(comments, start=1):
    print(f"{i}: {comment}")
