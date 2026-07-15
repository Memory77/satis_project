from itemadapter import ItemAdapter
import re
from scrapy.exceptions import DropItem
from datetime import datetime
import logging
from parsel import Selector  # Scrapy utilise Parsel pour le parsing HTML

class NewFilmsPipeline:
    def process_item(self, item, spider):
        logging.info(f"✅ Avant nettoyage : {item}")

         # ✅ Extraction du premier genre uniquement
        if "duree" in item and isinstance(item["duree"], str):
            selector = Selector(text=item["duree"])
            genres = selector.css("span.dark-grey-link::text").getall()
            item["genre"] = genres[0] if genres else None  # Prend uniquement le premier genre

        # Nettoyage général des valeurs vides
        for key in list(item.keys()):
            if item[key] == '-' or item[key] == []:
                item[key] = None

        # Nettoyage des champs spécifiques
        item['date_sortie'] = self.convert_date(item.get('date_sortie', ''))
        item['duree'] = self.clean_duration(item.get('duree', ''))
        
        if 'entrees' in item:
            item['entrees'] = self.convert_entrees(item['entrees'])

        if 'budget' in item:
            item['budget'] = self.convert_entrees(item['budget'])

        item['description'] = self.clean_text(item.get('description', ''))
        item['pays'] = item.get('pays', '').strip()
        
        if item.get('studio'):
            item['studio'] = item['studio'].strip()

        # ✅ Nettoyage du champ "anecdotes"
        if item.get('anecdotes'):
            item['anecdotes'] = item['anecdotes'].strip()
            premiere_lettre = item['anecdotes'][0]
            item['anecdotes'] = int(premiere_lettre)

        # ✅ Extraction du nombre de salles
        if 'salles' in item:
            item['salles'] = self.extract_sessions(item['salles'])

        # ✅ Nettoyage des champs "réalisateur" et "acteurs"
        if item.get('realisateur'):
            if item['realisateur'][0] == 'De':
                item['realisateur'].pop(0)
            item["realisateur"] = item["realisateur"][0] if item["realisateur"] else None



        if item.get('acteurs'):
            if item['acteurs'][0] == 'Avec':
                item['acteurs'].pop(0)

        logging.info(f"✅ Après nettoyage : {item}")
        return item

    # 🔹 Nettoyage de la durée (conversion en minutes)
    def clean_duration(self, duration_html):
        if duration_html:
            match = re.search(r'(\d+)h\s*(\d+)min', duration_html)
            if match:
                hours, minutes = match.groups()
                return int(hours) * 60 + int(minutes)
        return None

    # 🔹 Conversion du nombre d'entrées en int
    def convert_entrees(self, entrees):
        if entrees is None:
            return None
        if isinstance(entrees, int):
            return entrees
        if isinstance(entrees, str):
            return int(re.sub(r'\D', '', entrees))
        return None

    # 🔹 Suppression des espaces inutiles dans les descriptions
    def clean_text(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    # 🔹 Extraction du nombre de salles
    def extract_sessions(self, salles):
        if isinstance(salles, int):
            return salles
        elif isinstance(salles, str):
            match = re.search(r'\d+', salles)
            return int(match.group(0)) if match else None
        return None

    # 🔹 Conversion de la date en format ISO "YYYY-MM-DD"
    def convert_date(self, date_str):
        try:
            return datetime.strptime(date_str, '%d %B %Y').strftime('%Y-%m-%d')
        except ValueError:
            # Gestion des noms de mois en français
            french_to_english = {
                'janvier': 'January', 'février': 'February', 'mars': 'March', 'avril': 'April',
                'mai': 'May', 'juin': 'June', 'juillet': 'July', 'août': 'August',
                'septembre': 'September', 'octobre': 'October', 'novembre': 'November', 'décembre': 'December'
            }
            for fr, en in french_to_english.items():
                if fr in date_str:
                    date_str = date_str.replace(fr, en)
                    break
            try:
                return datetime.strptime(date_str, '%d %B %Y').strftime('%Y-%m-%d')
            except ValueError:
                return date_str  # Retourne la valeur d'origine si la conversion échoue


import mysql.connector
from mysql.connector import Error as MySQLError
from scrapy.exceptions import NotConfigured

class MySQLStorePipeline:
    def __init__(self, db_info):
        self.db_info = db_info

    @classmethod
    def from_crawler(cls, crawler):
        db_info = {
            'user': crawler.settings.get('MYSQL_USER'),
            'password': crawler.settings.get('MYSQL_PASSWORD'),
            'host': crawler.settings.get('MYSQL_HOST'),
            'database': crawler.settings.get('MYSQL_DATABASE')
        }
        if not all(db_info.values()):
            raise NotConfigured("Database configuration is missing.")
        return cls(db_info)

    def open_spider(self, spider):
        try:
            self.conn = mysql.connector.connect(**self.db_info)
            self.cursor = self.conn.cursor()

            # ❌ Vider la BDD à chaque lancement du Scrapy Spider
            self.clean_database()

        except MySQLError as e:
            spider.logger.error(f"Erreur de connexion à la base de données : {e}")
            raise

    def close_spider(self, spider):
        self.cursor.close()
        self.conn.close()

    def clean_database(self):
        """ Supprime toutes les données des tables et réinitialise les IDs """
        try:
            self.cursor.execute("DELETE FROM Participations;")
            self.cursor.execute("DELETE FROM films;")
            self.cursor.execute("DELETE FROM Personnes;")
            self.cursor.execute("ALTER TABLE films AUTO_INCREMENT = 1;")
            self.cursor.execute("ALTER TABLE Participations AUTO_INCREMENT = 1;")
            self.conn.commit()
            print("✅ Base de données nettoyée avant le crawl.")
        except MySQLError as e:
            print(f"❌ Erreur lors du nettoyage de la BDD: {e}")
            self.conn.rollback()

    def process_item(self, item, spider):
        film_id = self.insert_film(item)

        for role, people in [('acteur', item.get('acteurs', [])), ('réalisateur', item.get('realisateur', []))]:
            print(f"ROLE: {role}, PEOPLE: {people}")  # 🔍 Vérifier les valeurs avant de boucler

            if people:  # ✅ Vérifie que people n'est pas None avant de boucler
                for person in people:
                    person_id = self.ensure_person_exists(person)
                    self.link_person_to_film(film_id, person_id, role)

            return item

    def insert_film(self, item):
        query = """
        INSERT INTO films (titre, duree, salles, genre, date_sortie, pays, studio, description, image, budget, entrees, anecdotes) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
        duree = VALUES(duree), salles = VALUES(salles), genre = VALUES(genre), 
        date_sortie = VALUES(date_sortie), pays = VALUES(pays), studio = VALUES(studio),
        description = VALUES(description), image = VALUES(image), budget = VALUES(budget), 
        entrees = VALUES(entrees), anecdotes = VALUES(anecdotes);
        """
        self.cursor.execute(query, (
            item.get('titre'), item.get('duree'), item.get('salles'), item.get('genre'),
            item.get('date_sortie'), item.get('pays'), item.get('studio'), item.get('description'),
            item.get('image'), item.get('budget'), item.get('entrees'), item.get('anecdotes')
        ))
        self.conn.commit()
        print(f"✅ Film inséré : {item.get('titre')}")
        return self.cursor.lastrowid

    def ensure_person_exists(self, person_name):
        self.cursor.execute("SELECT id_personne FROM Personnes WHERE nom = %s", (person_name,))
        result = self.cursor.fetchone()
        if result:
            return result[0]
        self.cursor.execute("INSERT INTO Personnes (nom) VALUES (%s)", (person_name,))
        self.conn.commit()
        return self.cursor.lastrowid

    def link_person_to_film(self, film_id, person_id, role):
        query = """
        INSERT INTO Participations (id_film, id_personne, role) VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE role = VALUES(role);
        """
        self.cursor.execute(query, (film_id, person_id, role))
        self.conn.commit()
