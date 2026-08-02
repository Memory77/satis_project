import re
import time
import scrapy

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def js_text(driver, el) -> str:
    """Retourne textContent (même si pas 'visible' pour Selenium)."""
    try:
        txt = driver.execute_script("return arguments[0].textContent || '';", el)
        return (txt or "").strip()
    except Exception:
        return ""


def js_inner_text(driver, el) -> str:
    """Retourne innerText (plus proche de ce que tu vois à l'écran)."""
    try:
        txt = driver.execute_script("return arguments[0].innerText || '';", el)
        return (txt or "").strip()
    except Exception:
        return ""


def scroll_into_view(driver, el):
    try:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    except Exception:
        pass


def extract_rating(txt: str):
    """
    Sur Accor on voit souvent '4/5' ou '3.5/5' dans le bloc.
    Retourne float ou None.
    """
    if not txt:
        return None
    m = re.search(r"([0-5](?:[.,]5)?)/5", txt)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except Exception:
        return None


class AccorReviewsSpider(scrapy.Spider):
    name = "accor_reviews"
    allowed_domains = ["all.accor.com"]

    start_urls = [
        "https://all.accor.com/booking/fr/accor/hotels/nord-pas-de-calais-nord-pas-de-calais-france"
        "?dateIn=2026-02-11&nights=1&compositions=1&stayplus=false&snu=false&hideWDR=false"
        "&productCode=null&accessibleRooms=false&hideHotelDetails=false&filters="
    ]

    custom_settings = {
        "ROBOTSTXT_OBEY": False,
        "USER_AGENT": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        # "LOG_LEVEL": "INFO",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        opts = Options()
        opts.add_argument("--headless=new")   # enlève si tu veux voir le navigateur
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=opts)
        self.seen = set()

    def parse(self, response):
        d = self.driver
        d.get(response.url)

        # attendre les résultats
        WebDriverWait(d, 25).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.result-list-item__hotel-info"))
        )

        first_card = d.find_element(By.CSS_SELECTOR, "div.result-list-item__hotel-info")

        # titre (comme sur ton screenshot: span.ads-link__label)
        hotel_title = ""
        try:
            t = first_card.find_element(By.CSS_SELECTOR, "h2 a span.ads-link__label")
            hotel_title = js_inner_text(d, t)
        except Exception:
            pass

        # lien hôtel
        link = None
        for css in ["h2 a", "a.result-list-item__button"]:
            try:
                link = first_card.find_element(By.CSS_SELECTOR, css)
                break
            except Exception:
                continue
        if not link:
            return

        before = d.window_handles
        d.execute_script("arguments[0].click();", link)

        # switch onglet si ouverture nouvelle tab
        WebDriverWait(d, 20).until(lambda drv: len(drv.window_handles) >= len(before))
        after = d.window_handles
        if len(after) > len(before):
            new_tab = [h for h in after if h not in before][0]
            d.switch_to.window(new_tab)

        hotel_url = d.current_url

        WebDriverWait(d, 25).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # scroller un peu pour que la section Avis charge
        for _ in range(6):
            d.execute_script("window.scrollBy(0, document.body.scrollHeight/3);")
            time.sleep(0.8)

        # attendre au moins 1 bloc avis (ce que tu as dans le DOM)
        WebDriverWait(d, 25).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.client-review-block.clients-reviews__item"))
        )

        # cliquer "Voir tous les avis" (tu l'as bien dans tes logs)
        try:
            buttons = d.find_elements(By.CSS_SELECTOR, "button, a")
            for b in buttons:
                if "voir tous les avis" in js_inner_text(d, b).lower():
                    scroll_into_view(d, b)
                    time.sleep(0.2)
                    d.execute_script("arguments[0].click();", b)
                    time.sleep(2.0)
                    break
        except Exception:
            pass

        # IMPORTANT: attendre que le texte des avis soit réellement non vide
        def wait_reviews_have_text(timeout=20):
            end = time.time() + timeout
            while time.time() < end:
                cards = d.find_elements(By.CSS_SELECTOR, "div.client-review-block.clients-reviews__item")
                if cards:
                    sample = js_text(d, cards[0])
                    if sample and len(sample) > 5:
                        return True
                time.sleep(0.5)
            return False

        wait_reviews_have_text(timeout=25)

        # maintenant on collecte
        cards = d.find_elements(By.CSS_SELECTOR, "div.client-review-block.clients-reviews__item")

        # cliquer "Voir plus" si présent DANS chaque carte
        for c in cards:
            try:
                for el in c.find_elements(By.CSS_SELECTOR, "button, a"):
                    if js_inner_text(d, el).strip().lower() == "voir plus":
                        d.execute_script("arguments[0].click();", el)
                        time.sleep(0.15)
                        break
            except Exception:
                pass

        # re-lire après expansion
        cards = d.find_elements(By.CSS_SELECTOR, "div.client-review-block.clients-reviews__item")

        for c in cards:
            raw = js_text(d, c)
            if not raw:
                continue

            # texte avis (sur ton screenshot: span.ui-body-01.client-review-block__text)
            content = ""
            try:
                content_el = c.find_element(By.CSS_SELECTOR, "span.ui-body-01.client-review-block__text")
                content = js_inner_text(d, content_el)
            except Exception:
                pass

            # titre (sur ton screenshot: span.ui-heading-03)
            title = ""
            try:
                title_el = c.find_element(By.CSS_SELECTOR, "span.ui-heading-03")
                title = js_inner_text(d, title_el)
            except Exception:
                pass

            # auteur (souvent le nom est dans le bloc infos top)
            author = ""
            try:
                top = c.find_element(By.CSS_SELECTOR, ".client-review-block__infos-top")
                author = js_inner_text(d, top)
            except Exception:
                # fallback: chercher un prénom/nom possible dans raw
                author = ""

            rating = extract_rating(raw)

            # dédup
            key = (author.strip(), title.strip(), content.strip())
            if key in self.seen:
                continue
            self.seen.add(key)

            yield {
                "hotel_title": hotel_title,
                "hotel_url": hotel_url,
                "author": author,
                "rating": rating,
                "title": title,
                "content": content,
                "raw": raw,
            }

    def closed(self, reason):
        try:
            self.driver.quit()
        except Exception:
            pass
