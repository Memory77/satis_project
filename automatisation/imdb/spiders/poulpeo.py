import re
import scrapy
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def _safe_text(el):
    if not el:
        return None
    try:
        t = (el.text or "").strip()
        if t:
            return t
        tc = (el.get_attribute("textContent") or "").strip()
        return tc if tc else None
    except Exception:
        return None


def _first(parent, css):
    try:
        return parent.find_element(By.CSS_SELECTOR, css)
    except Exception:
        return None


def _attr(el, name: str):
    try:
        return el.get_attribute(name)
    except Exception:
        return None


def _parse_author_date(s: str | None):
    if not s:
        return (None, None)
    s = " ".join(s.split())
    m = re.search(r"Avis publié par\s+(.+?)\s+le\s+(\d{2}/\d{2}/\d{4})", s, flags=re.I)
    if not m:
        return (None, None)
    return (m.group(1).strip(), m.group(2).strip())


def _parse_rating(review_el):
    hidden = _first(review_el, 'div.hidden[itemprop="reviewRating"]')
    if not hidden:
        return None

    meta_val = _first(hidden, 'meta[itemprop="ratingValue"]')
    if meta_val:
        v = _attr(meta_val, "content")
        try:
            return int(float(v))
        except Exception:
            return None

    txt = _safe_text(hidden)
    if txt:
        m = re.search(r"(\d+)", txt)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


class PoulpeoAvisSpider(scrapy.Spider):
    name = "poulpeo_avis"

    custom_settings = {
        "ROBOTSTXT_OBEY": False,
        "USER_AGENT": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "LOG_LEVEL": "INFO",
        "FEED_EXPORT_ENCODING": "utf-8",
    }

    def __init__(self, url="https://www.poulpeo.com/avis/darty.htm", load_more=10, *args, **kwargs):
        """
        load_more:
          - 0 => clique jusqu’à disparition du bouton (avec max_clicks interne)
          - n>0 => clique au max n fois
        """
        super().__init__(*args, **kwargs)
        self.start_urls = [url]
        self.load_more = int(load_more)

        chrome_options = Options()
        chrome_options.add_argument("--headless=new")  # enlève si tu veux voir Chrome
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

        self.seen = set()

    def parse(self, response):
        self.driver.get(response.url)

        wait = WebDriverWait(self.driver, 25)

        # attendre 1er avis
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.review[itemprop="review"]')))

        max_clicks = self.load_more if self.load_more > 0 else 200  # sécurité
        clicks = 0

        def count_reviews():
            return len(self.driver.find_elements(By.CSS_SELECTOR, 'div.review[itemprop="review"]'))

        # cliquer "Voir plus..." tant que ça ajoute des avis
        while clicks < max_clicks:
            before = count_reviews()

            try:
                btn = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "div.plpLazyDataload-more button.plptable-more"))
                )
            except Exception:
                self.logger.info("Plus de bouton 'Voir plus...' (stop).")
                break

            try:
                self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
                self.driver.execute_script("arguments[0].click();", btn)
                clicks += 1
            except Exception:
                self.logger.info("Bouton trouvé mais click impossible (stop).")
                break

            # attendre que le nombre d’avis augmente (ou timeout => stop)
            try:
                wait.until(lambda d: count_reviews() > before)
            except Exception:
                after = count_reviews()
                self.logger.info(f"Clic #{clicks} mais aucun nouvel avis (before={before}, after={after}) => stop.")
                break

            after = count_reviews()
            self.logger.info(f"Clic #{clicks} OK: avis {before} -> {after}")

        # extraction
        reviews = self.driver.find_elements(By.CSS_SELECTOR, 'div.review[itemprop="review"]')
        self.logger.info(f"Total avis chargés: {len(reviews)}")

        for r in reviews:
            info = _safe_text(_first(r, "div.review-infos"))
            author, date_fr = _parse_author_date(info)
            content = _safe_text(_first(r, 'div.review-content[itemprop="reviewBody"]'))
            rating = _parse_rating(r)

            rid = f"{author}|{date_fr}|{content}"
            rid = f"poulpeo:{hash(rid)}"
            if rid in self.seen:
                continue
            self.seen.add(rid)

            yield {
                "source": "poulpeo",
                "page_url": response.url,
                "review_id": rid,
                "author": author,
                "date_text": date_fr,
                "rating": rating,
                "content": content,
            }

    def closed(self, reason):
        try:
            if self.driver:
                self.driver.quit()
        except Exception:
            pass
