import asyncio
from twscrape.api import API
from urllib.parse import urlparse

class TwitterScraper:
    def __init__(self):
        pass

    def extract_tweet_id_from_url(self, tweet_url):
        path_segments = urlparse(tweet_url).path.split('/')
        if len(path_segments) >= 4 and path_segments[2] == 'status':
            return path_segments[3]
        return None

    async def get_tweet_replies(self, tweet_url, limit=50):
        tweet_id = self.extract_tweet_id_from_url(tweet_url)
        if not tweet_id:
            return "Geçersiz X gönderi URL'si."

        api = API()
          # Veritabanındaki accountları yükle

        replies = []
        async for tweet in api.search(f"conversation_id:{tweet_id}", limit=limit):
            replies.append(tweet.rawContent)

        return replies

    async def main_replies(self, tweet_url):
        replies = await self.get_tweet_replies(tweet_url)
        if isinstance(replies, str):
            return replies
        else:
            result_text = f"{len(replies)} yorum bulundu:\n\n"
            for i, reply in enumerate(replies):
                result_text += f"- {reply}\n"
            return result_text

if __name__ == "__main__":
    async def run_scraper():
        scraper = TwitterScraper()
        tweet_url = input("X gönderi URL'sini girin: ").strip()
        output = await scraper.main_replies(tweet_url)
        print(output)

    asyncio.run(run_scraper())