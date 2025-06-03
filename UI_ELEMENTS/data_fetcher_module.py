import asyncio
from urllib.parse import urlparse, parse_qs

async def fetch_comments_from_youtube(video_url, comment_size, api_key, YOUTUBE_API_AVAILABLE_FLAG, build_youtube_service, clean_text_func):
    """
    Belirtilen YouTube video URL'sinden yorumları çeker.
    build_youtube_service: googleapiclient.discovery.build fonksiyonu
    clean_text_func: Metin temizleme fonksiyonu (preprocessing_module.clean_html_tags_and_time)
    """
    if not YOUTUBE_API_AVAILABLE_FLAG:
        print("DEBUG: YouTube API kullanılabilir değil.")
        return []
    if not api_key: # Sadece anahtarın boş olup olmadığını kontrol et
        print("DEBUG: YouTube API anahtarı boş (data_fetcher).")
        return []


    try:
        parsed_url = urlparse(video_url)
        video_id = None

        # Standart YouTube video URL'leri: youtube.com/watch?v=VIDEO_ID
        if parsed_url.hostname in ['youtube.com', 'www.youtube.com'] and parsed_url.path == '/watch':
            query_params = parse_qs(parsed_url.query)
            if 'v' in query_params:
                video_id = query_params['v'][0]
        # Kısaltılmış YouTube URL'leri: youtu.be/VIDEO_ID
        elif parsed_url.hostname == 'youtu.be':
            video_id = parsed_url.path.strip('/')
        # Gömülü (embed) URL'ler: youtube.com/embed/VIDEO_ID
        elif parsed_url.hostname in ['youtube.com', 'www.youtube.com'] and parsed_url.path.startswith('/embed/'):
            video_id = parsed_url.path.split('/')[2]

        if not video_id:
            print(f"DEBUG: YouTube video ID'si bulunamadı veya URL formatı tanınmadı: {video_url}")
            return []

        youtube = build_youtube_service("youtube", "v3", developerKey=api_key)
        comments = []
        next_page_token = None
        
        total_fetched_count = 0
        while total_fetched_count < comment_size:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, comment_size - total_fetched_count),
                pageToken=next_page_token
            )
            response = request.execute()
            
            # API'den gelen öğe yoksa döngüyü kır
            if not response.get("items"):
                print(f"DEBUG: YouTube API'den 'items' alanı boş geldi. Yanıt: {response}")
                if "error" in response:
                    print(f"DEBUG: YouTube API hata detayı: {response['error']}")
                break # Yorum bulunamadıysa veya hata varsa döngüyü sonlandır

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                clean_comment = clean_text_func(comment)
                if clean_comment:
                    comments.append(clean_comment)
            
            total_fetched_count = len(comments)
            next_page_token = response.get("nextPageToken")
            
            if not next_page_token:
                break
                
        return comments[:comment_size]
    except Exception as e:
        print(f"DEBUG: YouTube yorum çekme hatası (data_fetcher): {e}")
        return []

async def scrape_tweets_from_twitter(twitter_url, TWITTER_SCRAPER_AVAILABLE_FLAG, scraper_class):
    """
    Belirtilen Twitter URL'sinden tweet yanıtlarını çeker.
    scraper_class: x3.TwitterScraper sınıfı (bağımlılığı azaltmak için parametre olarak geçildi).
    Dönen sonuç bir liste (tweetler) veya hata mesajı (string) olabilir.
    """
    if not TWITTER_SCRAPER_AVAILABLE_FLAG:
        return "TwitterScraper kullanılamıyor."

    try:
        parsed_url = urlparse(twitter_url)
        if "twitter.com" not in parsed_url.netloc and "x.com" not in parsed_url.netloc:
             return "Lütfen geçerli bir Twitter/X URL'si girin."
        path_segments = parsed_url.path.split('/')
        if len(path_segments) < 4 or path_segments[2] != 'status':
            return "Şu anda sadece tekil tweet (status) URL'lerinin yanıtları desteklenmektedir. Lütfen '.../status/...' formatında bir URL girin."
    except Exception as e:
         return f"Geçersiz URL formatı: {e}"

    scraper = scraper_class()
    try:
        tweet_replies = await scraper.get_tweet_replies(tweet_url=twitter_url, limit=50)
        
        if isinstance(tweet_replies, str):
             return tweet_replies

        if not tweet_replies:
            return "Belirtilen URL'den tweet yanıtı çekilemedi."
            
        return tweet_replies
    except Exception as e:
        print(f"DEBUG: Tweet çekme veya işleme hatası (data_fetcher): {e}")
        return f"Tweet çekme veya işleme hatası: {e}"