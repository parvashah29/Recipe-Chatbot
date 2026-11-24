import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Hermetic config: no network, no LLM, no image folder, throwaway database.
_tmp = tempfile.mkdtemp()
os.environ["THEMEALDB_ENABLED"] = "0"
os.environ["GEMINI_API_KEY"] = ""
os.environ["RECIPE_IMAGE_DIR"] = os.path.join(_tmp, "does-not-exist")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_tmp, "test.db")
os.environ["SECRET_KEY"] = "test-secret"


@pytest.fixture(scope="session")
def app_module():
    import app as app_module
    app_module.app.config["TESTING"] = True
    return app_module


@pytest.fixture()
def client(app_module):
    return app_module.app.test_client()


@pytest.fixture()
def bot(app_module):
    app_module.chatbot.current_recipe = None
    app_module.chatbot.current_recipe_list = None
    return app_module.chatbot
