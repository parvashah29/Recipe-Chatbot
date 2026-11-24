import html
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from themealdb_client import TheMealDBClient
from llm_service import llm_service, recipe_scaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLACEHOLDER_URL = "/static/images/placeholder.svg"
MAX_LIST_RESULTS = 24  # cap on recipes rendered in one list


def esc(value) -> str:
    """HTML-escape any value for safe interpolation into generated markup."""
    return html.escape(str(value), quote=True)


def clean_ingredients(raw) -> List[str]:
    """Turn the scraped, newline/tab-riddled ingredient blob into clean lines."""
    if raw is None or (not isinstance(raw, str) and pd.isna(raw)):
        return []
    lines = []
    for part in str(raw).split('\n'):
        part = ' '.join(part.split())
        if not part:
            continue
        if part.startswith(',') and lines:      # ", roasted" continues previous item
            lines[-1] = f"{lines[-1]}{part}"
        else:
            lines.append(part)
    return lines


def split_steps(raw) -> List[str]:
    """Split an instruction paragraph into steps without breaking decimals like 1.5."""
    if raw is None or (not isinstance(raw, str) and pd.isna(raw)):
        return []
    text = ' '.join(str(raw).split())
    steps = re.split(r'(?<=[A-Za-z\)])\.\s*(?=[A-Z])', text)
    return [step.strip().rstrip('.').strip() for step in steps if step.strip().strip('.')]


def parse_minutes(value) -> float:
    """'Total in 35 M' -> 35. Unknown -> inf (never matches a 'under N minutes' filter)."""
    if isinstance(value, str):
        match = re.search(r'(\d+)', value)
        if match:
            return float(match.group(1))
    return float('inf')


class Chatbot:
    def __init__(self, csv_path=None, image_dir=None):
        """Initialize the chatbot with cuisine data.

        image_dir: folder with recipe photos. Defaults to $RECIPE_IMAGE_DIR or
        ./image_for_cuisines/data (git-ignored). A missing folder is fine: the UI
        falls back to the bundled static/images set and then to a placeholder.
        """
        csv_path = csv_path or os.path.join(BASE_DIR, 'cuisines.csv')
        self.image_dir = image_dir or os.getenv('RECIPE_IMAGE_DIR') or os.path.join(BASE_DIR, 'image_for_cuisines', 'data')
        self.static_image_dir = os.path.join(BASE_DIR, 'static', 'images')
        self._image_names = self._list_images()

        self.df = pd.read_csv(csv_path)
        self.stop_words = set(ENGLISH_STOP_WORDS)

        # Derive numeric cooking time (minutes) from the prep_time text.
        if 'cooking_time' not in self.df.columns:
            self.df['cooking_time'] = self.df['prep_time'].map(parse_minutes)
        # The dataset has no difficulty column; rate it from time so the filter means something.
        if 'difficulty' not in self.df.columns:
            self.df['difficulty'] = np.select(
                [self.df['cooking_time'] <= 30, self.df['cooking_time'] <= 60],
                ['easy', 'medium'], default='hard')

        self.recipe_number_mapping = {}

        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2,
            max_df=0.95
        )
        self.text_columns = ['name', 'description', 'cuisine', 'course', 'diet', 'ingredients', 'instructions']
        self.combined_text = self._prepare_text_data()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.combined_text)
        self.themealdb = TheMealDBClient()

        # Create response templates
        self.response_templates = {
            'greeting': [
                "Hello! I'm your cuisine assistant. How can I help you today?",
                "Hi there! I can help you discover delicious recipes and cuisines. What would you like to know?",
                "Hey! I'm here to help you explore different cuisines and recipes. What can I tell you about?"
            ],
            'farewell': [
                "Goodbye! Enjoy your culinary adventures!",
                "See you later! Happy cooking!",
                "Bye! Feel free to ask more questions about cuisines anytime!"
            ],
            'thanks': [
                "You're welcome! Let me know if you need more information about cuisines!",
                "No problem! I'm here to help with any cuisine-related questions!",
                "Glad to help! Feel free to ask more questions!"
            ],
            'unknown': [
                "I'm not sure about that. Could you try asking about specific cuisines, ingredients, or cooking methods?",
                "I don't understand. Try asking about recipes, cuisines, or cooking techniques.",
                "I'm not sure what you mean. You can ask about different cuisines, ingredients, or cooking methods."
            ]
        }
        
        # Store current recipe list for numbered requests
        self.current_recipe_list = None
        self.current_recipe = None  # Store the last shown recipe for follow-ups
        
        self.llm_service = llm_service
        self.recipe_scaler = recipe_scaler

    def _list_images(self):
        names = set()
        for folder in (self.image_dir, self.static_image_dir):
            try:
                names.update(os.listdir(folder))
            except OSError:
                pass
        return names

    def _prepare_text_data(self):
        """Prepare text data for TF-IDF vectorization."""
        combined_text = []
        for _, row in self.df.iterrows():
            text_parts = []
            for col in self.text_columns:
                if pd.notna(row[col]):
                    # Clean and normalize text
                    text = str(row[col]).lower()
                    text = re.sub(r'[^\w\s]', ' ', text)
                    text = ' '.join(text.split())  # Remove extra whitespace
                    text_parts.append(text)
            combined_text.append(' '.join(text_parts))
        return combined_text

    def _preprocess_query(self, query):
        """Lowercase, strip punctuation and stop words."""
        query = query.lower()
        query = re.sub(r'[^\w\s&+-]', ' ', query)
        tokens = [t for t in query.split() if t not in self.stop_words]
        return ' '.join(tokens)

    def _get_best_match(self, query, top_k=3):
        """Find the best matching recipes based on TF-IDF similarity."""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return top_indices, similarities[top_indices]

    def similar_indices(self, idx, top_k=5):
        """Row indices of the recipes most similar to row `idx` (TF-IDF cosine)."""
        sims = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sims[idx] = -1
        return np.argsort(sims)[-top_k:][::-1]

    # ------------------------------------------------------------------ images
    def _get_local_image_path(self, row):
        """URL for a recipe's photo, or a placeholder when the file is not available.

        Photos live in image_for_cuisines/data (not in git) with a bundled subset in
        static/images; the /static/images/<file> route serves from both.
        """
        try:
            name = row['image_file'] if 'image_file' in row.index else None
            if isinstance(name, str) and name in self._image_names:
                return f"/static/images/{name}"
        except Exception:
            pass
        return PLACEHOLDER_URL

    def recipe_to_dict(self, row) -> Dict[str, Any]:
        """JSON-safe dict (no NaN) for the frontend, with a stable 1-based id and image URL."""
        data = {}
        for key, value in row.to_dict().items():
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating, float)):
                value = None if (np.isnan(value) or np.isinf(value)) else float(value)
            elif value is not None and not isinstance(value, (str, int, bool)) and pd.isna(value):
                value = None
            data[key] = value
        data['ingredients'] = '\n'.join(clean_ingredients(row.get('ingredients')))
        data['id'] = int(row.name) + 1
        data['image'] = self._get_local_image_path(row)
        return data

    # ------------------------------------------------------------- formatting
    def _format_recipe_list(self, recipes_df):
        """Format a list of recipes grouped by cuisine and diet (capped for readability)."""
        total = len(recipes_df)
        recipes_df = recipes_df.head(MAX_LIST_RESULTS)
        cuisines = recipes_df['cuisine'].fillna('Other').unique()

        response = '<div class="recipe-list">'
        for cuisine in cuisines:
            cuisine_recipes = recipes_df[recipes_df['cuisine'].fillna('Other') == cuisine]
            response += '<div class="cuisine-section">'
            response += f'<h3 class="cuisine-title">{esc(cuisine)} Cuisine</h3>'

            diets = cuisine_recipes['diet'].fillna('Other').unique()
            for diet in diets:
                diet_recipes = cuisine_recipes[cuisine_recipes['diet'].fillna('Other') == diet]
                response += '<div class="diet-section">'
                response += f'<h4 class="diet-title">{esc(diet)}</h4>'
                response += '<div class="recipe-list-items">'
                for idx, recipe in diet_recipes.iterrows():
                    recipe_number = int(idx) + 1
                    image_path = self._get_local_image_path(recipe)
                    response += '<div class="recipe-list-item">'
                    response += f'<div class="recipe-number">{recipe_number}</div>'
                    response += (f'<div class="recipe-image"><img src="{esc(image_path)}" alt="{esc(recipe["name"])}" '
                                 f'class="recipe-thumbnail" loading="lazy" '
                                 f'onerror="this.onerror=null;this.src=\'{PLACEHOLDER_URL}\'"></div>')
                    response += '<div class="recipe-list-info">'
                    response += f'<strong>{esc(recipe["name"])}</strong>'
                    if pd.notna(recipe['course']):
                        response += f'<span class="recipe-course"> ({esc(recipe["course"])})</span>'
                    response += '</div></div>'
                    self.recipe_number_mapping[recipe_number] = idx
                response += '</div></div>'  # recipe-list-items, diet-section
            response += '</div>'  # cuisine-section

        if total > MAX_LIST_RESULTS:
            response += f'<p class="recipe-instruction">Showing {MAX_LIST_RESULTS} of {total} matches. Add more detail to narrow it down.</p>'
        response += '<p class="recipe-instruction"><strong>To view a complete recipe, type "give me recipe number X"</strong> (where X is the number shown next to the recipe)</p>'
        response += '</div>'

        self.current_recipe_list = recipes_df
        return response

    def _format_response(self, cuisine_data):
        """Format a single recipe as HTML."""
        def field(key, default=''):
            value = cuisine_data.get(key) if hasattr(cuisine_data, 'get') else None
            return default if value is None or (not isinstance(value, str) and pd.isna(value)) else value

        response = '<div class="recipe-details">'
        response += f'<h2 class="recipe-title">{esc(field("name", "Recipe"))}</h2>'
        image_path = self._get_local_image_path(cuisine_data)
        response += (f'<img src="{esc(image_path)}" alt="{esc(field("name"))}" class="recipe-header-image" '
                     f'style="max-width:100%;max-height:260px;border-radius:12px;object-fit:cover" '
                     f'onerror="this.onerror=null;this.src=\'{PLACEHOLDER_URL}\'">')

        response += '<div class="recipe-section"><h3 class="recipe-section-title">Basic Information:</h3><ul class="recipe-info-list">'
        response += f'<li><strong>Cuisine:</strong> {esc(field("cuisine", "Not specified"))}</li>'
        response += f'<li><strong>Course:</strong> {esc(field("course", "Not specified"))}</li>'
        response += f'<li><strong>Diet:</strong> {esc(field("diet", "Not specified"))}</li>'
        if field('prep_time'):
            response += f'<li><strong>Preparation Time:</strong> {esc(field("prep_time"))}</li>'
        response += '</ul></div>'

        if field('description'):
            response += ('<div class="recipe-section"><h3 class="recipe-section-title">Description:</h3>'
                         f'<p class="recipe-description">{esc(field("description"))}</p></div>')

        ingredients = clean_ingredients(field('ingredients', None))
        if ingredients:
            response += '<div class="recipe-section"><h3 class="recipe-section-title">Ingredients:</h3><ul class="ingredients-list">'
            response += ''.join(f'<li>{esc(i)}</li>' for i in ingredients)
            response += '</ul></div>'

        steps = split_steps(field('instructions', None))
        if steps:
            response += '<div class="recipe-section"><h3 class="recipe-section-title">Instructions:</h3><ol class="instructions-list">'
            response += ''.join(f'<li>{esc(s)}</li>' for s in steps)
            response += '</ol></div>'

        response += '</div>'
        return response

    # ----------------------------------------------------------------- search
    def fuzzy_ingredient_search(self, ingredient: str, threshold: int = 80) -> List[Any]:
        """Search for recipes using fuzzy matching on ingredient lines."""
        ingredient = ingredient.lower().strip()
        matches = []
        for _, recipe in self.df.iterrows():
            for line in clean_ingredients(recipe['ingredients']):
                line = line.lower()
                if ingredient in line or fuzz.partial_ratio(ingredient, line) >= max(threshold, 90):
                    matches.append(recipe)
                    break
        return matches

    def multi_ingredient_search(self, ingredients: List[str], require_all: bool = True) -> List[Any]:
        """Search for recipes containing multiple ingredients."""
        ingredients = [ing.lower() for ing in ingredients if ing]
        if not ingredients:
            return []
        text = self.df['ingredients'].fillna('').str.lower()
        found = pd.concat([text.str.contains(re.escape(i), regex=True) for i in ingredients], axis=1)
        mask = found.all(axis=1) if require_all else found.any(axis=1)
        return [row for _, row in self.df[mask].iterrows()]

    def search_by_cooking_time(self, max_time: int) -> pd.DataFrame:
        """Recipes that can be prepared within `max_time` minutes."""
        return self.df[self.df['cooking_time'] <= max_time]

    def search_by_difficulty(self, difficulty: str) -> pd.DataFrame:
        return self.df[self.df['difficulty'].str.lower() == difficulty.lower()]

    # ------------------------------------------------------------ query parsing
    COMMON_CUISINES = [
        'south indian', 'north indian', 'indian', 'italian', 'chinese', 'mexican', 'thai', 'japanese', 'french',
        'greek', 'spanish', 'moroccan', 'lebanese', 'turkish', 'american', 'british',
        'punjabi', 'bengali', 'gujarati', 'maharashtrian',
        'kerala', 'tamil', 'andhra', 'rajasthani', 'kashmiri', 'goan'
    ]
    COURSE_KEYWORDS = {
        'breakfast': ['breakfast', 'morning meal'],
        'lunch': ['lunch', 'midday meal'],
        'dinner': ['dinner', 'evening meal', 'supper'],
        'snack': ['snack', 'evening snack', 'tea time'],
        'dessert': ['dessert', 'sweet', 'pudding', 'ice cream'],
        'appetizer': ['appetizer', 'starter', 'first course'],
        'main course': ['main course', 'main dish', 'entree']
    }
    # key -> (keywords, regex on the diet column, regex that must NOT match)
    DIET_KEYWORDS = {
        'non vegetarian': (['non-vegetarian', 'non vegetarian', 'non veg', 'nonveg', 'non-veg'], r'non\s*veg', None),
        'vegan': (['vegan', 'plant based', 'plant-based'], r'vegan', None),
        'vegetarian': (['vegetarian', 'veg', 'veggie', 'no meat'], r'vegetarian', r'non\s*veg'),
        'gluten free': (['gluten free', 'gluten-free', 'no gluten'], r'gluten', None),
        'high protein': (['high protein', 'protein rich', 'protein-rich'], r'high protein', None),
    }

    @staticmethod
    def _has(text, phrase):
        return re.search(r'(?<![a-z])' + re.escape(phrase) + r'(?![a-z])', text) is not None

    def _detect_diet(self, text):
        for key, (words, _, _) in self.DIET_KEYWORDS.items():
            if any(self._has(text, w) for w in words):
                return key
        return None

    def _detect_course(self, text):
        for key, words in self.COURSE_KEYWORDS.items():
            if any(self._has(text, w) for w in words):
                return key
        return None

    def _detect_cuisine(self, text):
        for cuisine in self.COMMON_CUISINES:
            if self._has(text, cuisine):
                return cuisine
        return None

    def _detect_difficulty(self, text):
        mapping = {'easy': 'easy', 'simple': 'easy', 'beginner': 'easy', 'medium': 'medium',
                   'hard': 'hard', 'difficult': 'hard', 'advanced': 'hard'}
        for word, level in mapping.items():
            if self._has(text, word):
                return level
        return None

    def _local_filter(self, cuisine=None, course=None, diet=None, max_time=None, difficulty=None):
        """Apply every detected constraint at once to the local dataset."""
        df = self.df
        if cuisine:
            df = df[df['cuisine'].fillna('').str.lower().str.contains(re.escape(cuisine))]
        if course:
            df = df[df['course'].fillna('').str.lower().str.contains(re.escape(course))]
        if diet:
            _, include, exclude = self.DIET_KEYWORDS[diet]
            d = df['diet'].fillna('').str.lower()
            df = df[d.str.contains(include)]
            if exclude:
                df = df[~df['diet'].fillna('').str.lower().str.contains(exclude)]
        if max_time is not None:
            df = df[df['cooking_time'] <= max_time]
        if difficulty:
            df = df[df['difficulty'] == difficulty]
        return df

    def _describe(self, cuisine, course, diet, max_time, difficulty):
        parts = [p for p in (difficulty, diet, cuisine, course) if p]
        label = ' '.join(parts).title() or 'matching'
        if max_time is not None:
            label += f' under {max_time} minutes'
        return label

    def get_response(self, user_query, user_name=None):
        """Answer a user message. Local dataset first for filters; TheMealDB (if reachable) for
        cuisine/ingredient/name lookups; TF-IDF best match as the final fallback."""
        user_query = (user_query or '').strip()
        if not user_query:
            return "Please type a question, for example \"paneer recipe\" or \"vegetarian dinner under 30 minutes\"."
        user_query = user_query[:500]
        original_query = user_query.lower()
        processed_query = self._preprocess_query(user_query)

        # Greetings / small talk (whole words only, so "chicken" or "this" do not trigger them)
        if any(self._has(original_query, w) for w in ['hello', 'hi', 'hey']):
            if user_name:
                return f"Hello, {esc(user_name)}! How can I help you today?"
            return str(np.random.choice(self.response_templates['greeting']))
        elif any(self._has(original_query, w) for w in ['bye', 'goodbye', 'see you']):
            return str(np.random.choice(self.response_templates['farewell']))
        elif any(self._has(original_query, w) for w in ['thanks', 'thank you']):
            return str(np.random.choice(self.response_templates['thanks']))

        if self._has(original_query, 'help') or 'what can you do' in original_query:
            return self._get_help_info()

        # Numbered recipe request ("give me recipe number 3")
        number_match = re.search(r'(?:number|recipe|#)\s*#?\s*(\d+)\b(?!\s*(?:min|minute|serving|people|portion))', original_query)
        if number_match:
            recipe_number = int(number_match.group(1))
            if 1 <= recipe_number <= len(self.df):
                recipe = self.df.iloc[recipe_number - 1]
                self.current_recipe = recipe
                return self._format_response(recipe) + '<br><i>(From local recipe database)</i>'
            return f"Sorry, I couldn't find recipe number {recipe_number}. Please try a number between 1 and {len(self.df)}."

        # Follow-ups about the current recipe (health, substitutions, scaling, tips, similar)
        follow_up_response = self.process_follow_up(user_query, user_name)
        if follow_up_response:
            return follow_up_response

        # Combined local filters: diet / time / difficulty (+ cuisine, course when also present)
        time_match = re.search(r'(\d+)\s*(?:minute|minutes|min|mins)\b', original_query)
        max_time = int(time_match.group(1)) if time_match else None
        diet = self._detect_diet(original_query)
        difficulty = self._detect_difficulty(original_query)
        cuisine = self._detect_cuisine(original_query)
        course = self._detect_course(original_query)

        if max_time is not None or diet or difficulty:
            matches = self._local_filter(cuisine, course, diet, max_time, difficulty)
            label = self._describe(cuisine, course, diet, max_time, difficulty)
            if matches.empty:
                return f"I couldn't find any {label} recipes. Try loosening one of the filters."
            matches = matches.sort_values('cooking_time') if max_time is not None else matches
            return self._format_recipe_list(matches) + f'<br><i>(From local recipe database: {esc(label)} recipes)</i>'

        # Cuisine (+ optional course): TheMealDB first, then local
        if cuisine:
            api_results = self.themealdb.search_by_name(cuisine)
            if api_results:
                if course:
                    by_course = [m for m in api_results if course in str(m.get('strCategory', '')).lower()]
                    if by_course:
                        api_results = by_course
                names = ', '.join(esc(m['strMeal']) for m in api_results[:5])
                who = f"{esc(user_name)}, here" if user_name else "Here"
                return f"{who} are some {cuisine.title()} recipes from TheMealDB:<br>{names}<br><i>(From TheMealDB API)</i>"
            matches = self._local_filter(cuisine=cuisine, course=course)
            label = self._describe(cuisine, course, None, None, None)
            if not matches.empty:
                return self._format_recipe_list(matches) + f'<br><i>(From local recipe database: {esc(label)} recipes)</i>'
            return f"Sorry, I couldn't find any {label} recipes in my database. Try another cuisine or meal type."

        # Multi-ingredient search ("recipes with rice and beans")
        if re.search(r'\b(recipes?|dishes?|food|meals?)\b.*\b(with|containing|made with)\b', original_query) and (' and ' in original_query or ',' in original_query):
            tail = re.split(r'\b(?:made with|containing|with)\b', original_query, maxsplit=1)[1]
            ingredients = [i.strip(' .?!') for i in re.split(r',|\band\b', tail) if i.strip(' .?!')]
            api_meals = []
            for ing in ingredients:
                api_meals.extend(self.themealdb.search_by_ingredient(ing))
            if api_meals:
                names = ', '.join(esc(m['strMeal']) for m in api_meals[:5])
                who = f"{esc(user_name)}, here" if user_name else "Here"
                return f"{who} are recipes from TheMealDB with your ingredients:<br>{names}<br><i>(From TheMealDB API)</i>"
            matches = self.multi_ingredient_search(ingredients)
            if matches:
                return self._format_recipe_list(pd.DataFrame(matches)) + '<br><i>(From local recipe database)</i>'
            return f"I couldn't find any recipes containing {esc(', '.join(ingredients))}. Try other ingredients or a broader search."

        # Course only
        if course:
            matches = self._local_filter(course=course)
            if not matches.empty:
                return self._format_recipe_list(matches) + '<br><i>(From local recipe database)</i>'
            return f"I couldn't find any {course} recipes. Try another meal type."

        # Single-ingredient queries ("recipes with paneer")
        ingredient = None
        for pattern in (r'(?:made with|using|contains|containing|with)\s+([a-z][a-z ]+?)\s*[?.!]*$',):
            m = re.search(pattern, original_query)
            if m:
                ingredient = m.group(1).strip()
        if ingredient:
            api_results = self.themealdb.search_by_ingredient(ingredient)
            if api_results:
                names = ', '.join(esc(m['strMeal']) for m in api_results[:5])
                who = f"{esc(user_name)}, here" if user_name else "Here"
                return f"{who} are recipes from TheMealDB with {esc(ingredient)}:<br>{names}<br><i>(From TheMealDB API)</i>"
            matches = self.fuzzy_ingredient_search(ingredient)
            if matches:
                return self._format_recipe_list(pd.DataFrame(matches)) + '<br><i>(From local recipe database)</i>'
            return f"I couldn't find any recipes with {esc(ingredient)}. Try another ingredient."

        # Recipe name: TheMealDB first, then best TF-IDF match in the local dataset
        api_results = self.themealdb.search_by_name(user_query)
        if api_results:
            meal = api_results[0]
            self.current_recipe = None
            who = f"{esc(user_name)}, here's" if user_name else "Here's"
            return (f"{who} a recipe from TheMealDB!<br><b>{esc(meal.get('strMeal', ''))}</b><br>"
                    f"Category: {esc(meal.get('strCategory', ''))}<br>Instructions: {esc(meal.get('strInstructions', ''))}"
                    f"<br><i>(From TheMealDB API)</i>")

        top_indices, similarities = self._get_best_match(processed_query)
        if similarities[0] < 0.2:
            food_keywords = ['recipe', 'dish', 'food', 'cuisine', 'meal', 'cook', 'prepare']
            if any(self._has(original_query, k) for k in food_keywords):
                random_recipes = self.df.sample(min(10, len(self.df)))
                return ("I'm not sure exactly what you're looking for, but here are some recipes you might like:"
                        + self._format_recipe_list(random_recipes) + '<br><i>(From local recipe database)</i>')
            return str(np.random.choice(self.response_templates['unknown']))
        best_match = self.df.iloc[top_indices[0]]
        self.current_recipe = best_match
        response = self._format_response(best_match)
        if user_name:
            response = f"<b>{esc(user_name)}, here's a recipe you might like:</b><br>" + response
        return response + '<br><i>(From local recipe database)</i>'

    def _get_help_info(self):
        """Return help information about what the chatbot can do."""
        help_text = """
        <div class="help-info">
            <h3>How to Use the Cuisine Chatbot:</h3>
            <ul>
                <li><strong>Search by cuisine:</strong> "Show me Indian recipes" or "I want Italian food"</li>
                <li><strong>Search by ingredients:</strong> "Recipes with rice and beans" or "Dishes containing paneer"</li>
                <li><strong>Search by diet:</strong> "Show vegetarian recipes" or "Non-veg dishes"</li>
                <li><strong>Search by course:</strong> "Breakfast recipes" or "Show me some desserts"</li>
                <li><strong>Search by cooking time:</strong> "Quick recipes under 30 minutes"</li>
                <li><strong>Search by difficulty:</strong> "Easy recipes" or "Show me some advanced dishes"</li>
                <li><strong>View specific recipe:</strong> After seeing a list, type "Give me recipe number 3"</li>
            </ul>
            <p>Try asking about specific cuisines or ingredients you're interested in!</p>
        </div>
        """
        return help_text

    def _current_recipe_data(self) -> Optional[Dict[str, Any]]:
        """Current recipe as a plain dict (ingredients cleaned), or None."""
        if self.current_recipe is None:
            return None
        if isinstance(self.current_recipe, dict):
            return self.current_recipe
        data = self.current_recipe.to_dict()
        data = {k: (None if not isinstance(v, str) and pd.isna(v) else v) for k, v in data.items()}
        data['ingredients'] = '\n'.join(clean_ingredients(data.get('ingredients')))
        return data

    def get_health_explanation(self, recipe: Dict = None) -> str:
        """Health explanation (LLM if configured, rule-based otherwise)."""
        recipe_data = recipe if recipe is not None else self._current_recipe_data()
        if recipe_data is None:
            return "I don't have a recipe selected. Please search for a recipe first!"
        explanation = self.llm_service.explain_recipe_health(recipe_data)
        return f"""
        <div class="health-explanation">
            <div class="icon">🌿</div>
            <h4>Health Benefits of {esc(recipe_data.get('name') or 'this recipe')}:</h4>
            <p>{esc(explanation)}</p>
        </div>
        """

    def get_ingredient_substitutions(self, ingredient: str = None) -> str:
        """Ingredient substitutions for a named ingredient or the current recipe."""
        if not ingredient:
            recipe_data = self._current_recipe_data()
            if recipe_data is None:
                return "Tell me which ingredient to replace, e.g. \"substitute for butter\"."
            ingredients = (recipe_data.get('ingredients') or '').lower()
            common = ['eggs', 'egg', 'milk', 'butter', 'cream', 'paneer', 'chicken', 'rice', 'sugar', 'flour']
            found = [c for c in common if c in ingredients]
            if not found:
                return "I couldn't find any common ingredients to substitute in this recipe. Ask me about a specific ingredient!"
            ingredient = 'eggs' if found[0] == 'egg' else found[0]

        substitutions = self.llm_service.get_ingredient_substitutions(ingredient)
        response = f"""
        <div class="substitution-info">
            <h4>🔄 Substitutions for {esc(ingredient.title())}:</h4>
            <ul class="substitution-list">
        """
        for sub in substitutions:
            response += f"""
                <li>
                    <strong>{esc(sub['name'])}</strong>
                    <span class="sub-ratio">({esc(sub['ratio'])})</span>
                    <span class="sub-type badge">{esc(sub['type'])}</span>
                </li>
            """
        return response + "</ul></div>"

    def scale_recipe(self, target_servings: int = None, query: str = None) -> str:
        """Scale the current recipe's ingredient quantities (dataset recipes assume 4 servings)."""
        recipe_data = self._current_recipe_data()
        if recipe_data is None:
            return "Please select a recipe first before scaling!"
        if not target_servings and query:
            target_servings = self.recipe_scaler.extract_servings_from_query(query)
        if not target_servings:
            return "How many servings would you like? Try saying 'Make this for 8 people' or 'Scale to 6 servings'."
        target_servings = max(1, min(int(target_servings), 100))

        original_servings = 4
        scaled = self.recipe_scaler.scale_ingredients(
            recipe_data.get('ingredients') or '', original_servings, target_servings)

        response = f"""
        <div class="scaled-recipe">
            <h4>📐 {esc(recipe_data.get('name') or 'Recipe')} - Scaled for {target_servings} servings:</h4>
            <p class="scale-info">Original: {original_servings} servings → New: {target_servings} servings</p>
            <div class="scaled-ingredients">
                <h5>Adjusted Ingredients:</h5>
                <ul class="ingredients-list">
        """
        for ing in scaled.split('\n'):
            if ing.strip():
                response += f"<li>{esc(ing.strip())}</li>"
        return response + "</ul></div></div>"

    def get_cooking_tips(self) -> str:
        """Contextual cooking tips for the current recipe."""
        recipe_data = self._current_recipe_data()
        if recipe_data is None:
            return "Show me a recipe first, and I'll give you some cooking tips!"
        tips = self.llm_service.generate_cooking_tips(recipe_data)
        response = '<div class="cooking-tips"><h4>💡 Pro Tips for This Recipe:</h4><ul>'
        response += ''.join(f"<li>{esc(tip)}</li>" for tip in tips)
        return response + "</ul></div>"

    def get_similar_recipes(self) -> str:
        """Recipes similar to the current one (TF-IDF cosine similarity)."""
        if self.current_recipe is None or isinstance(self.current_recipe, dict):
            return "Please view a recipe first, then ask for similar recipes!"
        idx = self.current_recipe.name
        similar = self.df.iloc[self.similar_indices(idx, top_k=5)]
        if similar.empty:
            return "I couldn't find similar recipes. Try a different search!"
        return self._format_recipe_list(similar) + '<br><i>(Similar recipes from local database)</i>'

    def _build_json_response(self, response_text: str, recipe_data: Dict = None, source: str = None) -> Dict[str, Any]:
        """Build a structured JSON response for the frontend."""
        return {
            'response': response_text,
            'recipe': recipe_data,
            'source': source,
            'suggestions': self._get_follow_up_suggestions(recipe_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_follow_up_suggestions(self, recipe_data: Dict = None) -> List[Dict]:
        """Generate smart follow-up suggestions based on context."""
        suggestions = []
        
        if recipe_data:
            suggestions = [
                {'label': 'Want substitutions?', 'icon': '🔄', 'action': 'substitutions'},
                {'label': 'Similar recipes', 'icon': '👀', 'action': 'similar'},
                {'label': 'Health benefits', 'icon': '💚', 'action': 'healthy'},
                {'label': 'Scale recipe', 'icon': '📐', 'action': 'scale'},
                {'label': 'Cooking tips', 'icon': '💡', 'action': 'tips'},
            ]
        else:
            suggestions = [
                {'label': 'Show me Indian recipes', 'icon': '🍛', 'query': 'Show me Indian recipes'},
                {'label': 'Quick breakfast ideas', 'icon': '🌅', 'query': 'Quick breakfast recipes'},
                {'label': 'Vegetarian dishes', 'icon': '🥗', 'query': 'Vegetarian recipes'},
                {'label': 'Healthy dinner', 'icon': '🥦', 'query': 'Healthy dinner recipes'},
            ]
        
        return suggestions
    
    def process_follow_up(self, query: str, user_name: str = None) -> Optional[str]:
        """Handle follow-ups about the current recipe. Returns None when the query is not one."""
        q = query.lower()
        has_recipe = self.current_recipe is not None

        if any(w in q for w in ['substitut', 'replace', 'instead of', 'alternative']):
            m = re.search(r'(?:substitutes?|substitution|replace|replacement|alternatives?|instead of)\s+(?:for\s+|of\s+)?(?:the\s+)?(\w+)', q)
            ingredient = m.group(1) if m and m.group(1) not in ('this', 'it', 'ingredients', 'recipe') else None
            if ingredient or has_recipe:
                return self.get_ingredient_substitutions(ingredient)

        if not has_recipe:
            return None

        if any(self._has(q, w) for w in ['healthy', 'health', 'nutritious', 'benefits']) or 'good for' in q:
            return self.get_health_explanation()
        if any(self._has(q, w) for w in ['scale', 'servings', 'people', 'portions', 'double', 'half']):
            if self._has(q, 'double'):
                return self.scale_recipe(target_servings=8)
            if self._has(q, 'half'):
                return self.scale_recipe(target_servings=2)
            return self.scale_recipe(query=query)
        if any(w in q for w in ['similar', 'like this', 'more like', 'related']):
            return self.get_similar_recipes()
        if any(self._has(q, w) for w in ['tip', 'tips', 'advice', 'trick']):
            return self.get_cooking_tips()
        return None
