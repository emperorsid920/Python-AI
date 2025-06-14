from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_jwt_extended import JWTManager

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'FHIUHFI872EJ'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
    app.config['JWT_SECRET_KEY'] = 'super-secret-jwt-key'  # Change this!

    db.init_app(app)
    CORS(app, supports_credentials=True)
    JWTManager(app)

    from .api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    with app.app_context():
        db.create_all()

    return app