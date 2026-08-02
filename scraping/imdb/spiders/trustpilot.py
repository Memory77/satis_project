import re
import scrapy
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# --------- helpers selenium ----------
def _safe_text(el):
    try:
        t = el.text
        return t.strip() if t else None
    except Exception:
        return None


def _first(parent, css):
    try:
        return parent.find_element(By.CSS_SELECTOR, css)
    except Exception:
        return None


def _parse_rating_from_alt(alt_text: str | None):
    # ex: "Noté 5 sur 5 étoiles"
    if not alt_text:
        return None
    m = re.search(r"(\d)\s+sur\s+5", alt_text)
    return int(m.group(1)) if m else None


# --------- nettoyage ----------
STOP_LINES = {
    "Utile",
    "Partager",
    "Avis spontané",
    "Vérifié",
    "L'entreprise a répondu",
    "L’entreprise a répondu",
    "Voir plus",
}

def _fix_common_typos(s: str | None) -> str | None:
    if not s:
        return s
    # Sur certaines polices, "Il y a" ressort comme "ll y a"
    s = s.replace("ll y a", "Il y a")
    return s


def clean_raw(raw: str | None) -> str | None:
    if not raw:
        return raw

    raw = raw.replace("\r", "")
    lines = [l.strip() for l in raw.split("\n") if l.strip()]

    cleaned = []
    for l in lines:
        l = _fix_common_typos(l)

        # lignes parasites simples
        if l in STOP_LINES:
            continue

        # "•"
        if l == "•":
            continue

        # pays (FR, BE, etc.)
        if re.fullmatch(r"[A-Z]{2}", l):
            continue

        # "3 avis", "11 avis"
        if re.fullmatch(r"\d+\s+avis", l):
            continue

        # compteur utile "1" isolé (souvent entre Utile et Partager)
        if re.fullmatch(r"\d+", l):
            continue

        cleaned.append(l)

    return "\n".join(cleaned).strip() if cleaned else None


class TrustpilotAllPagesCleanSpider(scrapy.Spider):
    name = "trustpilot"
    allowed_domains = ["fr.trustpilot.com"]

    custom_settings = {
        "ROBOTSTXT_OBEY": False,
        "USER_AGENT": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
    }

    def __init__(self, domain="www.fnac.com", start_page=1, max_pages=0, *args, **kwargs):
        """
        domain: ex "www.fnac.com"
        start_page: page de départ (pratique pour reprendre)
        max_pages:
          - 0 => jusqu’à la fin (ou jusqu’au blocage)
          - n>0 => limite à n pages
        """
        super().__init__(*args, **kwargs)
        self.domain = domain
        self.start_page = int(start_page)
        self.max_pages = int(max_pages)

        url = f"https://fr.trustpilot.com/review/{self.domain}"
        if self.start_page > 1:
            url += f"?page={self.start_page}"
        self.start_urls = [url]

        chrome_options = Options()
        chrome_options.add_argument("--headless=new")  # enlève si tu veux voir Chrome
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

        self.seen_ids = set()

    def parse(self, response):
        self.driver.get(response.url)

        page_num = self.start_page
        pages_done = 0

        while True:
            # attendre avis
            try:
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "article"))
                )
            except Exception:
                self.logger.error(f"Timeout: avis non chargés (page {page_num})")
                return

            # Scraper la page courante
            articles = self.driver.find_elements(By.CSS_SELECTOR, "article")

            for a in articles:
                review_id = None
                try:
                    review_id = a.get_attribute("data-service-review-id")
                except Exception:
                    review_id = None

                raw_block = _safe_text(a)
                if not review_id and raw_block:
                    review_id = f"noid:{hash(raw_block)}"

                if review_id and review_id in self.seen_ids:
                    continue
                if review_id:
                    self.seen_ids.add(review_id)

                card = _first(a, '[data-testid="service-review-card-v2"]') or a

                author = _safe_text(_first(card, "[data-consumer-name-typography]"))
                if author:
                    author = _fix_common_typos(author)

                time_el = _first(card, "time")
                date_iso = time_el.get_attribute("datetime") if time_el else None
                date_text = _safe_text(time_el) if time_el else None
                date_text = _fix_common_typos(date_text)

                title = _safe_text(_first(card, "[data-service-review-title-typography]")) or _safe_text(_first(card, "h2"))
                content = _safe_text(_first(card, "[data-service-review-text-typography]")) or _safe_text(_first(card, "p"))
                title = _fix_common_typos(title)
                content = _fix_common_typos(content)

                rating = None
                rating_el = _first(card, "div[data-service-review-rating]")
                if rating_el:
                    try:
                        rating = rating_el.get_attribute("data-service-review-rating")
                        rating = int(rating) if rating else None
                    except Exception:
                        rating = None

                if rating is None:
                    img_star = _first(card, "img[alt*='sur 5']")
                    if img_star:
                        rating = _parse_rating_from_alt(img_star.get_attribute("alt"))

                raw_clean = clean_raw(raw_block)

                # fallback author : 1ère ligne du raw nettoyé
                if not author and raw_clean:
                    lines = [l.strip() for l in raw_clean.split("\n") if l.strip()]
                    if lines:
                        author = lines[0]

                # ✅ un seul yield => pas de doublons
                yield {
                    "company_domain": self.domain,
                    "page": page_num,
                    "review_id": review_id,
                    "author": author,
                    "rating": rating,
                    "date_iso": date_iso,
                    "date_text": date_text,
                    "title": title,
                    "content": content,
                    "raw_clean": raw_clean,
                    "url": self.driver.current_url,
                }

            pages_done += 1
            if self.max_pages > 0 and pages_done >= self.max_pages:
                return

            # next ?
            next_link = None
            for css in ['a[name="pagination-button-next"]', 'a[rel="next"]']:
                try:
                    next_link = self.driver.find_element(By.CSS_SELECTOR, css)
                    if next_link:
                        break
                except Exception:
                    next_link = None

            if not next_link:
                return

            current_url = self.driver.current_url
            try:
                next_link.click()
                WebDriverWait(self.driver, 20).until(lambda d: d.current_url != current_url)
            except Exception:
                href = None
                try:
                    href = next_link.get_attribute("href")
                except Exception:
                    href = None
                if not href:
                    return
                self.driver.get(href)

            page_num += 1

    def closed(self, reason):
        try:
            if self.driver:
                self.driver.quit()
        except Exception:
            pass