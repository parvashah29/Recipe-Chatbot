# Recipe Chatbot: Indian cuisine and world recipes

A Flask chatbot that answers recipe questions from a local dataset of ~4,200 mostly Indian recipes (`cuisines.csv`), with optional lookups on TheMealDB and optional Gemini-written health notes. Retrieval uses TF-IDF similarity plus rule-based filters. It needs no GPU, no model downloads and no API keys.

## What works

- **Chat search**: "paneer recipe", "Show me Indian recipes", "breakfast recipes", "recipes with rice and beans", "easy vegan recipes". Filters combine, so "vegetarian dinner under 30 minutes" applies diet, course and time together. Cooking time is parsed from the dataset's `prep_time`; difficulty is derived from that time (the dataset has no difficulty field).
- **Recipe detail**: "give me recipe number 12" (numbers are shown in result lists).
- **Follow-ups about the last recipe**: health notes, ingredient substitutions, scaling ("make this for 8 people"; dataset recipes assume 4 servings), cooking tips, similar recipes (TF-IDF cosine).
- **Accounts**: register / login / logout, favourites (`POST /favorite` toggles, `GET /favorites` lists), comments (`POST /comment`, `GET /comments/<id>`), recent chat history on `/homepage`. JSON endpoints: `GET /recipe/<id>`, `GET /health`.
- **TheMealDB** (free, no key) is tried first for cuisine, ingredient and name queries. If the network is down it is skipped silently and the local data answers.
- **Gemini** (optional) writes the "health benefits" text. Without a key a rule-based explanation is used. Models are tried in a fallback chain with retries on 429/5xx.

## Limitations

- Conversation context (last recipe / last list) is held in one server-side object shared by all users, so the app is meant for single-user or demo use.
- The Gemini path is covered by a mocked fallback-chain test only; it has not been run against the live API.
- The browser UI (theme toggle, sharing buttons, etc. in `static/js/chat.js`) was not exercised in a real browser. Backend behaviour is what the tests cover.

## Setup

Requires Python 3.10+ (tested on 3.12, macOS arm64).

```bash
git clone https://github.com/parvashah29/Recipe-Chatbot.git
cd Recipe-Chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # optional; edit to taste
python app.py             # http://127.0.0.1:5001
```

Startup builds the TF-IDF index (a few seconds). The SQLite database is created automatically in `instance/`.

### Configuration (all optional, see `.env.example`)

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Flask session key. Random per start if unset (logins reset on restart). |
| `GEMINI_API_KEY` | Enables Gemini health notes. Keep it in `.env`, which is git-ignored. |
| `GEMINI_MODELS` | Comma-separated model fallback chain (default `gemini-3-flash-preview,gemini-3.1-flash-lite,...`). |
| `THEMEALDB_ENABLED` | `0` disables network lookups (offline mode). |
| `RECIPE_IMAGE_DIR` | Folder of recipe photos. |
| `DATABASE_URL`, `HOST`, `PORT`, `FLASK_DEBUG` | Usual Flask settings. |

### Recipe images

The full photo set (~840 MB) is **not** in this repository. The app looks for photos in `image_for_cuisines/data/` (or `RECIPE_IMAGE_DIR`), then in the ~640 photos bundled in `static/images/`. Any recipe whose file is in neither place shows a placeholder picture, and the page layout is unaffected. A fresh clone therefore works, with about a quarter of recipes showing real photos. To get all photos, put the files named in the `image_file` column of `cuisines.csv` into `image_for_cuisines/data/`. The tests run with the image folder absent.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite uses Flask's test client with the network, LLM and image folder disabled. It covers chat queries, combined filters, recipe detail and follow-ups, bad input, registration/login/favourites/comments, missing images, and the Gemini fallback chain (mocked).

## Layout

```
app.py              Flask routes, auth, database models
chatbot.py          Query parsing, filtering, TF-IDF retrieval, HTML formatting
llm_service.py      Optional Gemini client (google-genai), substitutions, tips, scaler
themealdb_client.py TheMealDB client (timeouts, offline switch)
recipe_utils.py     Small helper module (not used by the app)
cuisines.csv        Recipe dataset
templates/, static/ UI
tests/              pytest suite
```
