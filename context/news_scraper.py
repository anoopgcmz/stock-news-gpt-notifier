import feedparser
from newspaper import Article

# Default RSS feed for Indian financial news.
YAHOO_FINANCE_INDIA_RSS = "https://in.finance.yahoo.com/rss/topstories"


def fetch_articles(rss_url: str = YAHOO_FINANCE_INDIA_RSS):
    """Fetch the latest financial news articles.

    Parameters
    ----------
    rss_url: str, optional
        RSS feed URL pointing to an Indian financial news source. Defaults to
        Yahoo Finance India's top stories feed.

    Returns
    -------
    list[dict]
        A list of dictionaries containing the article title and full text.
    """

    feed = feedparser.parse(rss_url)
    news = []
    for entry in feed.entries[:5]:
        article = Article(entry.link)
        article.download()
        article.parse()
        news.append({
            "title": article.title,
            "content": article.text,
        })
    return news
