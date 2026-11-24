import os
import requests


class TheMealDBClient:
    """Thin, failure-tolerant client for TheMealDB's free API.

    Every method returns a list (empty on any network/HTTP/parse problem) so the
    chatbot can silently fall back to the local dataset. Set THEMEALDB_ENABLED=0
    to skip network calls entirely (offline mode).
    """
    BASE_URL = "https://www.themealdb.com/api/json/v1/1/"
    TIMEOUT = 5

    def __init__(self):
        self.enabled = os.getenv("THEMEALDB_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")

    def _get(self, endpoint, params):
        if not self.enabled:
            return []
        try:
            response = requests.get(self.BASE_URL + endpoint, params=params, timeout=self.TIMEOUT)
            if response.status_code == 200:
                return response.json().get('meals') or []
        except (requests.RequestException, ValueError):
            pass
        return []

    def search_by_name(self, name):
        return self._get("search.php", {"s": name})

    def search_by_ingredient(self, ingredient):
        return self._get("filter.php", {"i": ingredient})

    def lookup_by_id(self, meal_id):
        return self._get("lookup.php", {"i": meal_id})

    def list_categories(self):
        return self._get("list.php", {"c": "list"})
