# Import necessary modules and classes
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager

# Initialize the database
db = SQLAlchemy()
# Naming the database
DB_NAME = "database.db"


# Function to create the Flask application
def create_app():
    # Initialize the Flask application
    app = Flask(__name__)

    # Set the secret key for the app
    app.config['SECRET_KEY'] = 'FHIUHFI872EJ'

    # Configure the location of the database
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'

    # Initialize the database with the app
    db.init_app(app)

    # Import blueprints for views, authentication, and models
    from .views import views
    from .auth import auth
    from .models import User, Note  # Adjust the import path as needed

    # Register blueprints with the app, specifying URL prefixes
    app.register_blueprint(views, url_prefix='/')

    app.register_blueprint(auth, url_prefix='/')

    # Create the database using the create_database function
    create_database(app)

    # Initialize the LoginManager
    login_manager = LoginManager()
    # Set the login view for redirecting unauthenticated users
    login_manager.login_view = 'auth.login'
    # Attach the LoginManager to the app
    login_manager.init_app(app)

    # Define a function to load a user by ID
    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    # Return the configured app
    return app


# Function to check if a model exists already and create the database if not
def create_database(app):
    # Import User and Note models to avoid circular import
    from .models import User, Note

    # Check if the database file exists
    if not path.exists('website/' + DB_NAME):
        # Use app context to create the database tables
        with app.app_context():
            db.create_all()
            print('Created Database!')
