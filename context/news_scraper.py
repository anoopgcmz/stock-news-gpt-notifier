import os

import feedparser
from newspaper import Article

# RSS feeds for Indian financial news sources.
YAHOO_FINANCE_INDIA_RSS = "https://in.finance.yahoo.com/rss/topstories"
ECONOMIC_TIMES_RSS = (
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"
)


def fetch_articles(rss_url: str | None = None):
    """Fetch the latest financial news articles.

    Parameters
    ----------
    rss_url: str, optional
        RSS feed URL pointing to an Indian financial news source. If not
        provided, the ``RSS_FEED_URL`` environment variable is used. If the
        environment variable is unset, defaults to the Economic Times RSS feed.

    Returns
    -------
    list[dict]
        A list of dictionaries containing the article title and full text.
    """

    feed_url = rss_url or os.getenv("RSS_FEED_URL", ECONOMIC_TIMES_RSS)
    feed = feedparser.parse(feed_url)
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
