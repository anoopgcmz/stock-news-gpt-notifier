import feedparser
from newspaper import Article

def fetch_articles(rss_url="https://finance.yahoo.com/news/rssindex"):
    feed = feedparser.parse(rss_url)
    news = []
    for entry in feed.entries[:5]:
        article = Article(entry.link)
        article.download()
        article.parse()
        news.append({
            "title": article.title,
            "content": article.text
        })
    return news
