from urllib.parse import urlparse


trusted_domains = [

    "bbc.com",
    "reuters.com",
    "thehindu.com",
    "indiatoday.in",
    "ndtv.com",
    "cnn.com",
    "nytimes.com"

]


def check_source_credibility(url):

    try:

        domain = urlparse(url).netloc.lower()

        for trusted in trusted_domains:

            if trusted in domain:

                return "Trusted Source"

        return "Unverified Source"

    except:

        return "Unknown Source"