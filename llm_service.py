"""
LLM Service for Recipe Chatbot
Integrates with Google Gemini API for natural conversations,
recipe explanations, and intelligent responses.
"""

import os
import re
import time
from typing import Dict, List, Optional
from fractions import Fraction

# Optional LLM: the app works fully without it (rule-based fallbacks everywhere).
MODEL_CHAIN = [m.strip() for m in os.getenv(
    "GEMINI_MODELS",
    "gemini-3-flash-preview,gemini-3.1-flash-lite,gemini-2.5-flash-lite,gemini-2.0-flash",
).split(",") if m.strip()]
MAX_RETRIES = 2


class LLMService:
    def __init__(self):
        """Gemini wrapper (google-genai SDK). Disabled when GEMINI_API_KEY is absent."""
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.client = None
        if self.is_configured():
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:  # SDK missing or bad config -> run without LLM
                print(f"LLM disabled: {type(e).__name__}")
                self.client = None

    def is_configured(self) -> bool:
        """True when an API key is present and the client could be built."""
        if self.client is not None:
            return True
        return len(self.api_key) > 10 and self.api_key != "YOUR_GEMINI_API_KEY_HERE"

    def _generate(self, prompt: str) -> Optional[str]:
        """Try each model in the chain with retries on transient errors. None on failure."""
        if self.client is None:
            return None
        for model in MODEL_CHAIN:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    resp = self.client.models.generate_content(model=model, contents=prompt)
                    text = (resp.text or "").strip()
                    if text:
                        return text
                    break  # empty answer: try next model
                except Exception as e:
                    msg = str(e)
                    transient = any(c in msg for c in ("503", "429", "500", "UNAVAILABLE", "overloaded"))
                    if transient and attempt < MAX_RETRIES:
                        time.sleep(0.5 * (2 ** attempt))
                        continue
                    break  # 404 (retired model), auth, etc: next model
        return None

    def generate_response(self, prompt: str, context: str = "") -> Optional[str]:
        """Generate a free-form response, or None if the LLM is unavailable."""
        full_prompt = f"{context}\n\nUser: {prompt}" if context else prompt
        return self._generate(full_prompt)

    def explain_recipe_health(self, recipe: Dict) -> str:
        """Generate health explanation for a recipe."""
        prompt = f"""
            Analyze this recipe and explain its health benefits in a friendly, conversational way.
            Keep the response concise (3-4 sentences) but informative.
            
            Recipe: {recipe.get('name', 'Unknown')}
            Ingredients: {recipe.get('ingredients', 'Not available')}
            Cuisine: {recipe.get('cuisine', 'Unknown')}
            Diet: {recipe.get('diet', 'Not specified')}
            
            Focus on:
            1. Key nutritional benefits
            2. Healthy ingredients highlighted
            3. Any health considerations
            """
        return self._generate(prompt) or self._fallback_health_explanation(recipe)
    
    def _fallback_health_explanation(self, recipe: Dict) -> str:
        """Fallback health explanation when LLM is not available."""
        diet = str(recipe.get('diet') or '').lower()
        ingredients = str(recipe.get('ingredients') or '').lower()
        
        health_points = []
        
        if 'vegetarian' in diet:
            health_points.append("🥬 This vegetarian dish is rich in plant-based nutrients")
        if 'vegan' in diet:
            health_points.append("🌱 Vegan recipe - great for heart health and the environment")
        if any(spice in ingredients for spice in ['turmeric', 'ginger', 'garlic']):
            health_points.append("🌿 Contains anti-inflammatory spices with medicinal properties")
        if any(protein in ingredients for protein in ['lentil', 'dal', 'paneer', 'chicken', 'fish']):
            health_points.append("💪 Good source of protein for muscle health")
        if any(veggie in ingredients for veggie in ['spinach', 'tomato', 'carrot', 'onion']):
            health_points.append("🥗 Packed with essential vitamins and minerals")
            
        if not health_points:
            return "🍽️ This recipe provides a balanced combination of nutrients. Enjoy in moderation as part of a healthy diet!"
            
        return " ".join(health_points[:3])
    
    def get_ingredient_substitutions(self, ingredient: str, recipe_context: str = "") -> List[Dict]:
        """Get smart ingredient substitutions."""
        # Common substitutions database
        substitutions = {
            'eggs': [
                {'name': 'Flax eggs', 'ratio': '1 tbsp ground flax + 3 tbsp water per egg', 'type': 'vegan'},
                {'name': 'Chia eggs', 'ratio': '1 tbsp chia seeds + 3 tbsp water per egg', 'type': 'vegan'},
                {'name': 'Applesauce', 'ratio': '1/4 cup per egg', 'type': 'vegan'},
                {'name': 'Mashed banana', 'ratio': '1/4 mashed banana per egg', 'type': 'vegan'},
            ],
            'milk': [
                {'name': 'Almond milk', 'ratio': '1:1 replacement', 'type': 'dairy-free'},
                {'name': 'Oat milk', 'ratio': '1:1 replacement', 'type': 'dairy-free'},
                {'name': 'Coconut milk', 'ratio': '1:1 replacement', 'type': 'dairy-free'},
                {'name': 'Soy milk', 'ratio': '1:1 replacement', 'type': 'dairy-free'},
            ],
            'butter': [
                {'name': 'Coconut oil', 'ratio': '1:1 replacement', 'type': 'vegan'},
                {'name': 'Olive oil', 'ratio': '3/4 cup per 1 cup butter', 'type': 'vegan'},
                {'name': 'Ghee', 'ratio': '1:1 replacement', 'type': 'lactose-free'},
                {'name': 'Avocado', 'ratio': '1:1 replacement for baking', 'type': 'vegan'},
            ],
            'cream': [
                {'name': 'Coconut cream', 'ratio': '1:1 replacement', 'type': 'dairy-free'},
                {'name': 'Cashew cream', 'ratio': '1:1 replacement', 'type': 'vegan'},
                {'name': 'Greek yogurt', 'ratio': '1:1 replacement', 'type': 'lower-fat'},
            ],
            'paneer': [
                {'name': 'Tofu (firm)', 'ratio': '1:1 replacement', 'type': 'vegan'},
                {'name': 'Halloumi', 'ratio': '1:1 replacement', 'type': 'alternative'},
                {'name': 'Cottage cheese', 'ratio': '1:1 replacement', 'type': 'lower-fat'},
            ],
            'chicken': [
                {'name': 'Tofu', 'ratio': '1:1 replacement', 'type': 'vegan'},
                {'name': 'Tempeh', 'ratio': '1:1 replacement', 'type': 'vegan'},
                {'name': 'Jackfruit', 'ratio': '1:1 replacement (shredded)', 'type': 'vegan'},
                {'name': 'Seitan', 'ratio': '1:1 replacement', 'type': 'vegan'},
            ],
            'rice': [
                {'name': 'Cauliflower rice', 'ratio': '1:1 replacement', 'type': 'low-carb'},
                {'name': 'Quinoa', 'ratio': '1:1 replacement', 'type': 'protein-rich'},
                {'name': 'Brown rice', 'ratio': '1:1 replacement', 'type': 'whole-grain'},
            ],
            'sugar': [
                {'name': 'Honey', 'ratio': '3/4 cup per 1 cup sugar', 'type': 'natural'},
                {'name': 'Maple syrup', 'ratio': '3/4 cup per 1 cup sugar', 'type': 'vegan'},
                {'name': 'Stevia', 'ratio': '1 tsp per 1 cup sugar', 'type': 'zero-calorie'},
                {'name': 'Jaggery', 'ratio': '1:1 replacement', 'type': 'traditional'},
            ],
            'flour': [
                {'name': 'Almond flour', 'ratio': '1:1 replacement', 'type': 'gluten-free'},
                {'name': 'Coconut flour', 'ratio': '1/4 cup per 1 cup flour', 'type': 'gluten-free'},
                {'name': 'Oat flour', 'ratio': '1:1 replacement', 'type': 'whole-grain'},
            ],
        }
        
        ingredient_lower = ingredient.lower().strip()
        
        # Check for exact or partial matches
        for key, subs in substitutions.items():
            if key in ingredient_lower or ingredient_lower in key:
                return subs
        
        # Default response if no match
        return [{'name': 'No specific substitution found', 'ratio': 'Try similar ingredients', 'type': 'general'}]
    
    def get_similar_recipes_prompt(self, recipe: Dict) -> str:
        """Generate a smart prompt for finding similar recipes."""
        return f"Show me more {recipe.get('cuisine', '')} recipes" if recipe.get('cuisine') else "Show me similar recipes"
    
    def generate_cooking_tips(self, recipe: Dict) -> List[str]:
        """Generate contextual cooking tips for a recipe."""
        tips = []
        ingredients = str(recipe.get('ingredients') or '').lower()
        cuisine = str(recipe.get('cuisine') or '').lower()
        
        # Ingredient-based tips
        if 'rice' in ingredients:
            tips.append("💡 Rinse rice 2-3 times before cooking for fluffier results")
        if 'paneer' in ingredients:
            tips.append("💡 Soak paneer in warm water for 10 mins before cooking for softer texture")
        if 'chicken' in ingredients:
            tips.append("💡 Marinate chicken for at least 30 mins for better flavor absorption")
        if 'onion' in ingredients:
            tips.append("💡 Caramelize onions on low heat for richer flavor")
        if any(spice in ingredients for spice in ['cumin', 'coriander', 'turmeric']):
            tips.append("💡 Toast whole spices in oil before adding other ingredients")
            
        # Cuisine-based tips
        if 'indian' in cuisine:
            tips.append("💡 For authentic taste, use fresh curry leaves and mustard seeds")
        if 'italian' in cuisine:
            tips.append("💡 Al dente pasta continues cooking in the sauce - drain slightly early")
        if 'chinese' in cuisine:
            tips.append("💡 High heat and quick cooking preserves vegetable crunch")
            
        return tips[:3] if tips else ["💡 Read through all steps before starting to cook"]
    
    def enhance_response(self, query: str, base_response: str, recipe_data: Optional[Dict] = None) -> str:
        """Enhance a response with LLM if available, otherwise return base response."""
        prompt = f"""
            You are a friendly, knowledgeable cooking assistant. Enhance this response to be more 
            conversational and helpful. Keep it concise but warm.
            
            User asked: {query}
            Base response: {base_response}
            
            Add a brief, friendly touch while keeping all the important information.
            """
        return self._generate(prompt) or base_response


class RecipeScaler:
    """Utility class for scaling recipe quantities."""
    
    # Common units and their conversions
    UNITS = {
        'cup': ['cup', 'cups', 'c'],
        'tablespoon': ['tablespoon', 'tablespoons', 'tbsp', 'tbs'],
        'teaspoon': ['teaspoon', 'teaspoons', 'tsp'],
        'ounce': ['ounce', 'ounces', 'oz'],
        'pound': ['pound', 'pounds', 'lb', 'lbs'],
        'gram': ['gram', 'grams', 'g', 'gm'],
        'kilogram': ['kilogram', 'kilograms', 'kg'],
        'milliliter': ['milliliter', 'milliliters', 'ml'],
        'liter': ['liter', 'liters', 'l'],
        'piece': ['piece', 'pieces', 'pcs'],
        'whole': ['whole'],
        'pinch': ['pinch', 'pinches'],
        'bunch': ['bunch', 'bunches'],
        'clove': ['clove', 'cloves'],
        'sprig': ['sprig', 'sprigs'],
    }
    
    @staticmethod
    def parse_quantity(qty_str: str) -> Optional[float]:
        """Parse a quantity string to a float."""
        qty_str = qty_str.strip()
        
        # Handle fractions like "1/2", "1/4"
        if '/' in qty_str:
            try:
                parts = qty_str.split()
                if len(parts) == 2:  # "1 1/2" format
                    whole = float(parts[0])
                    frac = float(Fraction(parts[1]))
                    return whole + frac
                else:
                    return float(Fraction(qty_str))
            except:
                pass
        
        # Handle ranges like "2-3"
        if '-' in qty_str:
            try:
                parts = qty_str.split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            except:
                pass
        
        # Handle simple numbers
        try:
            return float(qty_str)
        except:
            return None
    
    @staticmethod
    def format_quantity(qty: float) -> str:
        """Format a quantity nicely (convert decimals to fractions where appropriate)."""
        if qty == int(qty):
            return str(int(qty))
        
        # Common fractions
        fractions = {
            0.25: '1/4',
            0.33: '1/3',
            0.5: '1/2',
            0.67: '2/3',
            0.75: '3/4',
        }
        
        whole = int(qty)
        decimal = qty - whole
        
        # Find closest fraction
        for dec, frac in fractions.items():
            if abs(decimal - dec) < 0.1:
                if whole > 0:
                    return f"{whole} {frac}"
                return frac
        
        # Round to 1 decimal place if no fraction match
        if whole > 0:
            return f"{round(qty, 1)}"
        return f"{round(qty, 2)}"
    
    @classmethod
    def scale_ingredients(cls, ingredients: str, original_servings: int, target_servings: int) -> str:
        """Scale ingredients from original servings to target servings."""
        if original_servings == target_servings or original_servings <= 0:
            return ingredients
        
        scale_factor = target_servings / original_servings
        
        # Pattern to match quantities at the beginning of ingredient lines
        # Matches: "2 cups", "1/2 tsp", "1 1/2 tablespoons", etc.
        quantity_pattern = r'^(\d+(?:\s+\d+)?(?:/\d+)?|\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*'
        
        lines = ingredients.split('\n') if '\n' in ingredients else ingredients.split(',')
        scaled_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            match = re.match(quantity_pattern, line)
            if match:
                qty_str = match.group(1)
                qty = cls.parse_quantity(qty_str)
                
                if qty is not None:
                    scaled_qty = qty * scale_factor
                    formatted_qty = cls.format_quantity(scaled_qty)
                    scaled_line = line[:match.start()] + formatted_qty + ' ' + line[match.end():]
                    scaled_lines.append(scaled_line)
                else:
                    scaled_lines.append(line)
            else:
                scaled_lines.append(line)
        
        return '\n'.join(scaled_lines) if '\n' in ingredients else ', '.join(scaled_lines)
    
    @classmethod
    def extract_servings_from_query(cls, query: str) -> Optional[int]:
        """Extract target servings from a user query."""
        # Patterns like "for 8 people", "serves 6", "8 servings"
        patterns = [
            r'for\s+(\d+)\s*(?:people|persons|guests|servings)?',
            r'serves?\s+(\d+)',
            r'(\d+)\s*servings?',
            r'(\d+)\s*portions?',
            r'feed\s+(\d+)',
        ]
        
        query_lower = query.lower()
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return int(match.group(1))
        
        return None


# Initialize global instances
llm_service = LLMService()
recipe_scaler = RecipeScaler()
