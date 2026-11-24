import os
import re
import secrets
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)
from flask_login import (LoginManager, UserMixin, current_user, login_required,
                         login_user, logout_user)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv()

from chatbot import Chatbot  # noqa: E402  (after load_dotenv so env-driven config is seen)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


app = Flask(__name__, static_url_path='/static')
# Set SECRET_KEY in .env for sessions that survive restarts; otherwise a random key is used.
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY') or secrets.token_hex(32)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///recipes.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Image folder is optional (git-ignored, ~840 MB). See README.
chatbot = Chatbot()
IMAGE_DIR = chatbot.image_dir
STATIC_IMAGE_DIR = os.path.join(BASE_DIR, 'static', 'images')


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    comments = db.relationship('Comment', backref='author', lazy=True)
    favorites = db.relationship('Favorite', backref='user', lazy=True)


class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=_utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recipe_id = db.Column(db.Integer, nullable=False)


class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recipe_id = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=_utcnow)


with app.app_context():
    db.create_all()


def _json_body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _recipe_id(value):
    """Validate a 1-based recipe id; None when it is not an existing recipe."""
    try:
        rid = int(value)
    except (TypeError, ValueError):
        return None
    return rid if 1 <= rid <= len(chatbot.df) else None


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None

# Routes
@app.route('/')
def chat_page():
    return render_template('index.html')

@app.route('/homepage')
def homepage():
    chat_history = []
    if current_user.is_authenticated:
        # Load chat history from session or database (for demo, use session)
        chat_history = session.get('chat_history', [])
    return render_template('home.html', chat_history=chat_history)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = _json_body()
        username = str(data.get('username') or '').strip()
        password = str(data.get('password') or '')
        user = User.query.filter_by(username=username).first() if username else None
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = _json_body()
        username = str(data.get('username') or '').strip()
        email = str(data.get('email') or '').strip().lower()
        password = str(data.get('password') or '')
        if not username or len(username) > 80:
            return jsonify({'success': False, 'message': 'A username (max 80 characters) is required'}), 400
        if not re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+', email) or len(email) > 120:
            return jsonify({'success': False, 'message': 'A valid email is required'}), 400
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists'}), 409
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': 'Email already registered'}), 409

        user = User(username=username, email=email, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        return jsonify({'success': True})
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('chat_page'))


@app.route('/chat', methods=['POST'])
def chat():
    user_message = _json_body().get('message', '')
    if not isinstance(user_message, str) or not user_message.strip():
        return jsonify({'error': 'message is required'}), 400
    user_name = current_user.username if current_user.is_authenticated else None
    response = chatbot.get_response(user_message, user_name=user_name)
    
    # Get follow-up suggestions based on current context
    current_recipe = None
    if chatbot.current_recipe is not None:
        current_recipe = (chatbot.current_recipe if isinstance(chatbot.current_recipe, dict)
                          else chatbot.recipe_to_dict(chatbot.current_recipe))
    
    suggestions = chatbot._get_follow_up_suggestions(current_recipe)
    
    # Determine data source from response
    source = None
    if 'TheMealDB API' in response:
        source = 'TheMealDB API'
    elif 'local recipe database' in response:
        source = 'Local Database'
    
    return jsonify({
        'response': response,
        'suggestions': suggestions,
        'source': source,
        'recipe': current_recipe
    })

@app.route('/favorite', methods=['POST'])
@login_required
def favorite_recipe():
    recipe_id = _recipe_id(_json_body().get('recipe_id'))
    if recipe_id is None:
        return jsonify({'success': False, 'message': 'A valid recipe ID is required'}), 400

    existing = Favorite.query.filter_by(user_id=current_user.id, recipe_id=recipe_id).first()
    if existing:
        db.session.delete(existing)
        db.session.commit()
        return jsonify({'success': True, 'action': 'removed'})

    db.session.add(Favorite(user_id=current_user.id, recipe_id=recipe_id))
    db.session.commit()
    return jsonify({'success': True, 'action': 'added'})


@app.route('/favorites')
@login_required
def list_favorites():
    favs = Favorite.query.filter_by(user_id=current_user.id).order_by(Favorite.timestamp.desc()).all()
    return jsonify({'favorites': [
        {'recipe_id': f.recipe_id, 'name': str(chatbot.df.iloc[f.recipe_id - 1]['name'])}
        for f in favs if 1 <= f.recipe_id <= len(chatbot.df)]})


@app.route('/comment', methods=['POST'])
@login_required
def add_comment():
    data = _json_body()
    recipe_id = _recipe_id(data.get('recipe_id'))
    text = str(data.get('text') or '').strip()
    if recipe_id is None or not text:
        return jsonify({'success': False, 'message': 'Recipe ID and comment text required'}), 400

    comment = Comment(text=text[:2000], recipe_id=recipe_id, user_id=current_user.id)
    db.session.add(comment)
    db.session.commit()
    return jsonify({
        'success': True,
        'comment': {
            'id': comment.id,
            'text': comment.text,
            'author': comment.author.username,
            'timestamp': comment.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }
    })


@app.route('/recipe/<int:recipe_id>')
def recipe_detail(recipe_id):
    """JSON detail for one recipe (1-based row number, as shown in recipe lists)."""
    if _recipe_id(recipe_id) is None:
        abort(404)
    return jsonify(chatbot.recipe_to_dict(chatbot.df.iloc[recipe_id - 1]))


@app.route('/comments/<int:recipe_id>')
def get_comments(recipe_id):
    comments = Comment.query.filter_by(recipe_id=recipe_id).order_by(Comment.timestamp.desc()).all()
    return jsonify({
        'comments': [{
            'id': c.id,
            'text': c.text,
            'author': c.author.username,
            'timestamp': c.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        } for c in comments]
    })

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    """Serve recipe photos from the (optional, git-ignored) image folder, then the bundled
    static/images subset. Missing files return 404 and the UI swaps in the placeholder."""
    for folder in (IMAGE_DIR, STATIC_IMAGE_DIR):
        if os.path.isdir(folder):
            try:
                return send_from_directory(folder, filename)
            except Exception:  # NotFound / bad path: try next folder
                continue
    abort(404)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'recipes': int(len(chatbot.df)),
                    'llm': chatbot.llm_service.is_configured(),
                    'images_dir_present': os.path.isdir(IMAGE_DIR)})


@app.route('/save_chat', methods=['POST'])
def save_chat():
    if not current_user.is_authenticated:
        return ('', 204)
    chat_history = session.get('chat_history', [])
    data = _json_body()
    chat_history.append({
        'message': data.get('message', ''),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
    })
    session['chat_history'] = chat_history[-20:]  # Keep only last 20 messages
    return ('', 204)

if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG', '0') == '1',
            host=os.getenv('HOST', '127.0.0.1'),
            port=int(os.getenv('PORT', '5001')))
