import time
import re
import scrapy

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def try_get_text(root, css: str) -> str:
    try:
        el = root.find_element(By.CSS_SELECTOR, css)
        return clean_text(el.text)
    except Exception:
        return ""


def try_get_attr(root, css: str, attr: str) -> str:
    try:
        el = root.find_element(By.CSS_SELECTOR, css)
        v = el.get_attribute(attr) or ""
        return clean_text(v)
    except Exception:
        return ""


class BookingReviewsSpider(scrapy.Spider):
    name = "booking_reviews"
    allowed_domains = ["booking.com"]

    custom_settings = {
        # évite de paralléliser (Selenium = 1 navigateur)
        "CONCURRENT_REQUESTS": 1,
        "DOWNLOAD_TIMEOUT": 120,
        "LOG_LEVEL": "INFO",
    }

    def __init__(self, url=None, max_reviews=200, headless=1, *args, **kwargs):
        """
        url: URL de l'hôtel Booking
        max_reviews: limite de reviews à extraire
        headless: 1/0
        """
        super().__init__(*args, **kwargs)
        self.start_url = url or "https://www.booking.com/hotel/fr/le-flobart.fr.html"
        self.max_reviews = int(max_reviews)
        self.headless = str(headless) not in ("0", "false", "False")

    def start_requests(self):
        yield scrapy.Request(self.start_url, callback=self.parse, dont_filter=True)

    def _build_driver(self):
        opts = Options()
        if self.headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1400,900")
        # optionnel: réduit un peu le bruit
        opts.add_argument("--disable-gpu")
        return webdriver.Chrome(options=opts)

    def _accept_cookies_if_any(self, driver):
        # Les sélecteurs cookies changent souvent => on tente plusieurs boutons connus
        candidates = [
            "button#onetrust-accept-btn-handler",
            "button[data-testid='cookie-policy-dialog-accept-button']",
            "button:has(span:contains('Accepter'))",  # peut ne pas marcher selon Chrome/Selenium
        ]
        for css in candidates:
            try:
                btn = driver.find_element(By.CSS_SELECTOR, css)
                btn.click()
                time.sleep(0.5)
                return
            except Exception:
                pass

    def _open_reviews_modal(self, driver, wait):
        """
        Objectif: afficher la modale 'Commentaires clients' (overlay)
        """
        # 1) Essayer de cliquer l’onglet reviews (ton screenshot montre data-testid="Property-Header-Nav-Tab-Trigger-reviews")
        try:
            tab = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a#reviews-tab-trigger, a[data-testid*='Trigger-reviews']"))
            )
            driver.execute_script("arguments[0].click();", tab)
            time.sleep(1.0)
        except Exception:
            # pas bloquant, parfois la section est déjà visible
            pass

        # 2) Cliquer le lien/bouton qui ouvre tous les commentaires
        # Booking varie: "Lire tous les commentaires", "Voir tous les commentaires", etc.
        openers = [
            "a[data-testid='Property-Header-Nav-Tab-Trigger-reviews']",
            "a#reviews-tab-trigger",
            "a[href*='#tab-reviews']",
            "button[data-testid='see-all-reviews']",
            "a[data-testid='see-all-reviews-link']",
        ]

        clicked = False
        for css in openers:
            try:
                el = driver.find_element(By.CSS_SELECTOR, css)
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                time.sleep(0.5)
                driver.execute_script("arguments[0].click();", el)
                clicked = True
                time.sleep(1.0)
                break
            except Exception:
                pass

        # 3) Attendre la modale (dialog)
        dialog = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='dialog']")))
        return dialog, clicked

    def _scroll_reviews_modal(self, driver, dialog, target_count: int):
        """
        Scroll infini dans la modale jusqu’à atteindre target_count (ou blocage).
        """
        last_count = 0
        stable_rounds = 0

        while True:
            cards = dialog.find_elements(By.CSS_SELECTOR, "div[role='group'][aria-label='Commentaire']")
            count = len(cards)

            if count >= target_count:
                return

            # Si plus rien ne charge après quelques tours, on s’arrête
            if count == last_count:
                stable_rounds += 1
            else:
                stable_rounds = 0
                last_count = count

            if stable_rounds >= 6:
                return

            # scroll dans la modale
            try:
                driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", dialog)
            except Exception:
                # fallback: scroll global
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            time.sleep(1.0)

    def parse(self, response):
        driver = self._build_driver()
        wait = WebDriverWait(driver, 25)

        try:
            driver.get(response.url)
            time.sleep(2.0)
            self._accept_cookies_if_any(driver)

            # ouvrir la modale reviews
            dialog, _ = self._open_reviews_modal(driver, wait)

            # scroller pour charger un maximum (limité)
            self._scroll_reviews_modal(driver, dialog, self.max_reviews)

            # relire toutes les cards
            cards = dialog.find_elements(By.CSS_SELECTOR, "div[role='group'][aria-label='Commentaire']")
            cards = cards[: self.max_reviews]

            hotel_name = ""
            try:
                hotel_name = clean_text(driver.find_element(By.CSS_SELECTOR, "h2, h1").text)
            except Exception:
                pass

            for idx, card in enumerate(cards, start=1):
                # Score (ex: 6.0 / 7.0 / 8.0 ...)
                score = try_get_text(card, "div[data-testid='review-score']")

                # Titre (ex: "Bien", "Agréable")
                title = try_get_text(card, "h3, div[data-testid='review-title']")

                # Date (ex: "Commentaire envoyé le 29 septembre 2025")
                date_text = try_get_text(card, "span[data-testid='review-date'], div[data-testid='review-date']")

                # Positif / négatif (dans ton screenshot: data-testid="review-positive-text")
                positive = try_get_text(card, "div[data-testid='review-positive-text']")
                negative = try_get_text(card, "div[data-testid='review-negative-text']")

                # Auteur (souvent dans un bloc “Personne ayant écrit le commentaire”)
                author = ""
                country = ""
                try:
                    author_block = card.find_element(By.CSS_SELECTOR, "div[aria-label*='Personne ayant écrit']")
                    author = try_get_text(author_block, "span, div")
                    # parfois pays séparé
                    country = try_get_text(author_block, "span[data-testid='reviewer-country']")
                except Exception:
                    # fallback très large
                    author = try_get_text(card, "span[data-testid='reviewer-name']")
                    country = try_get_text(card, "span[data-testid='reviewer-country']")

                # Nettoyage final : si author est un gros blob, on garde le début
                author = clean_text(author)
                if len(author) > 80:
                    author = author[:80].rstrip() + "…"

                yield {
                    "source": "booking.com",
                    "hotel_name": hotel_name,
                    "hotel_url": response.url,
                    "idx_on_page": idx,
                    "score": score,
                    "title": title,
                    "date_text": date_text,
                    "positive": positive,
                    "negative": negative,
                    "author": author,
                    "country": country,
                    "raw": clean_text(card.text),
                }

        finally:
            driver.quit()
