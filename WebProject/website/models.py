# This is a sample Python script.
from website import db                  # Importing the db object from the website package
from flask_login import UserMixin       # Importing UserMixin for user login functionality
from sqlalchemy.sql import func         # Importing func from sqlalchemy.sql

# Database model for storing notes
class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)                         # Unique identifier for each note
    data = db.Column(db.String(10000))                                   # Data associated with the notes, limited to 10000 characters
    date = db.Column(db.DateTime(timezone=True), default=func.now())     # Timestamp for when the note was created, using the current time
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))            # Reference to the User model using foreign key relationship

# Database model definition for users
# All the user information will be stored here
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)     # Unique identifier for each user
    email = db.Column(db.String(150), unique=True)   # Unique email address for each user, limited to 150 characters
    password = db.Column(db.String(150))             # Stores the password for each user, limited to 150 characters
    first_name = db.Column(db.String(150))           # Stores the first name of each user, limited to 150 characters
    notes = db.relationship('Note')                  # List that will store all related notes for a user
