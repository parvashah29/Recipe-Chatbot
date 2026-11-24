import json


def chat(client, message):
    r = client.post("/chat", json={"message": message})
    assert r.status_code == 200, r.data
    return r.get_json()


def test_pages_render(client):
    for path in ("/", "/login", "/register", "/homepage"):
        assert client.get(path).status_code == 200, path


def test_health_without_images_or_llm(client):
    data = client.get("/health").get_json()
    assert data["status"] == "ok" and data["recipes"] > 4000
    assert data["llm"] is False and data["images_dir_present"] is False


def test_paneer_recipe_returns_recipe(client, bot):
    data = chat(client, "paneer recipe")
    assert "paneer" in data["response"].lower()
    assert data["recipe"] and data["recipe"]["id"] >= 1
    json.dumps(data, allow_nan=False)  # no NaN leaks into JSON
    assert "(From local recipe database)" in data["response"]


def test_vegetarian_dinner_under_30_minutes(client, bot):
    data = chat(client, "vegetarian dinner under 30 minutes")
    assert "recipe-list" in data["response"]
    shown = bot.current_recipe_list
    assert len(shown) > 0
    assert (shown["cooking_time"] <= 30).all()
    assert shown["diet"].str.lower().str.contains("vegetarian").all()
    assert not shown["diet"].str.lower().str.contains("non").any()
    assert shown["course"].str.lower().str.contains("dinner").all()


def test_cuisine_browse(client):
    assert "recipe-list" in chat(client, "Show me Indian recipes")["response"]


def test_course_and_diet_and_help(client):
    assert "recipe-list" in chat(client, "breakfast recipes")["response"]
    assert "recipe-list" in chat(client, "show vegan recipes")["response"] or "couldn't" in chat(client, "show vegan recipes")["response"]
    assert "How to Use" in chat(client, "help")["response"]


def test_word_boundary_greeting(client):
    # "chicken"/"this" contain "hi" but are not greetings
    assert "assistant" not in chat(client, "chicken curry")["response"].lower()
    assert "help you" in chat(client, "hi")["response"].lower() or "assistant" in chat(client, "hi")["response"].lower()


def test_list_is_capped(client):
    body = chat(client, "vegetarian recipes")["response"]
    assert body.count('class="recipe-list-item"') <= 24


def test_recipe_number_detail_and_followups(client, bot):
    data = chat(client, "give me recipe number 5")
    assert "recipe-details" in data["response"]
    assert data["recipe"]["id"] == 5
    assert "health-explanation" in chat(client, "is it healthy?")["response"]
    assert "scaled-recipe" in chat(client, "make this for 8 people")["response"]
    assert "cooking-tips" in chat(client, "any tips?")["response"]
    assert "recipe-list" in chat(client, "show me similar recipes")["response"]
    assert "substitution-info" in chat(client, "substitute for butter")["response"]


def test_recipe_number_out_of_range(client):
    assert "couldn't find recipe number" in chat(client, "recipe number 999999")["response"]


def test_followup_without_recipe_does_not_crash(client, bot):
    assert chat(client, "Healthy dinner recipes")["response"]
    bot.current_recipe = None
    assert chat(client, "scale to 6 servings")["response"]


def test_gibberish(client):
    assert chat(client, "zzzqqq xkcd")["response"]


def test_bad_input(client):
    assert client.post("/chat", json={}).status_code == 400
    assert client.post("/chat", data="not json", content_type="text/plain").status_code == 400
    assert client.post("/chat", json={"message": 5}).status_code == 400
    assert client.post("/login", data="x", content_type="text/plain").status_code == 401
    assert client.post("/register", json={"username": "a"}).status_code == 400
    assert client.post("/chat", json={"message": "<script>alert(1)</script> " * 200}).status_code == 200


def test_missing_images_degrade_gracefully(client, bot):
    r = client.get("/static/images/nope-not-there.jpg")
    assert r.status_code == 404
    assert client.get("/static/images/placeholder.svg").status_code == 200
    body = chat(client, "Show me Indian recipes")["response"]
    assert "placeholder.svg" in body or "/static/images/" in body
    # recipe_to_dict always yields an image URL (placeholder when file is absent)
    row = bot.df.iloc[0]
    assert bot.recipe_to_dict(row)["image"].startswith("/static/images/")


def test_recipe_json_endpoint(client):
    assert client.get("/recipe/1").status_code == 200
    assert client.get("/recipe/99999999").status_code == 404


def test_auth_favorites_comments_flow(client):
    assert client.post("/favorite", json={"recipe_id": 1}).status_code in (302, 401)
    r = client.post("/register", json={"username": "tester", "email": "t@example.com", "password": "secret1"})
    assert r.get_json()["success"] is True
    assert client.post("/register", json={"username": "tester", "email": "t2@example.com", "password": "secret1"}).status_code == 409
    assert client.post("/login", json={"username": "tester", "password": "wrong"}).status_code == 401
    assert client.post("/login", json={"username": "tester", "password": "secret1"}).get_json()["success"] is True

    assert chat(client, "hello")["response"].startswith("Hello, tester")
    assert client.post("/favorite", json={"recipe_id": 3}).get_json()["action"] == "added"
    assert client.post("/favorite", json={"recipe_id": "abc"}).status_code == 400
    favs = client.get("/favorites").get_json()["favorites"]
    assert favs and favs[0]["recipe_id"] == 3
    assert client.post("/favorite", json={"recipe_id": 3}).get_json()["action"] == "removed"

    c = client.post("/comment", json={"recipe_id": 3, "text": "yum"}).get_json()
    assert c["success"] and c["comment"]["author"] == "tester"
    assert client.get("/comments/3").get_json()["comments"][0]["text"] == "yum"
    assert client.post("/comment", json={"recipe_id": 3, "text": " "}).status_code == 400

    assert client.post("/save_chat", json={"message": "hi"}).status_code == 204
    assert "hi" in client.get("/homepage").get_data(as_text=True)
    assert client.get("/logout").status_code == 302


def test_llm_helpers_work_without_key():
    from llm_service import LLMService, RecipeScaler
    svc = LLMService()
    assert svc.is_configured() is False
    assert svc.generate_response("hi") is None
    assert svc.get_ingredient_substitutions("paneer")[0]["name"] == "Tofu (firm)"
    assert svc.explain_recipe_health({"diet": "Vegan", "ingredients": "lentil"})
    assert RecipeScaler.scale_ingredients("2 cups rice\n1/2 tsp salt", 4, 8) == "4 cups rice\n1 tsp salt"


def test_llm_fallback_chain(monkeypatch):
    from llm_service import LLMService
    import llm_service as mod
    svc = LLMService()
    calls = []

    class Models:
        def generate_content(self, model, contents):
            calls.append(model)
            if model == mod.MODEL_CHAIN[0]:
                raise RuntimeError("503 UNAVAILABLE")
            if model == mod.MODEL_CHAIN[1]:
                raise RuntimeError("404 model retired")
            return type("R", (), {"text": "ok"})()

    svc.client = type("C", (), {"models": Models()})()
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    assert svc._generate("x") == "ok"
    assert calls[:mod.MAX_RETRIES + 1] == [mod.MODEL_CHAIN[0]] * (mod.MAX_RETRIES + 1)
    assert calls[-1] == mod.MODEL_CHAIN[2]


def test_themealdb_offline_returns_empty():
    from themealdb_client import TheMealDBClient
    assert TheMealDBClient().search_by_name("x") == []
