from newspaper import Article


def extract_news_from_url(url):

    try:

        article = Article(url)

        article.download()

        article.parse()

        return article.title + " " + article.text

    except:

        return None