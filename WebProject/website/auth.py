# Import necessary modules and classes from Flask
from flask import Blueprint, render_template, request, flash, redirect, url_for

# Import the User model, password hashing functions, and the database instance from the website package
from website.models import User
from werkzeug.security import generate_password_hash, check_password_hash
from website.models import db

# Import Flask-Login functions
from flask_login import login_user, login_required, logout_user, current_user

# Define a Blueprint named 'auth'
auth = Blueprint('auth', __name__)

# Define the route for the login page
@auth.route('/login', methods=['GET', 'POST'])
def login():
    data = request.form  # Retrieve the data sent to the forms

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user:
            if check_password_hash(user.password, password):
                flash(f'Logged in successfully, {user.first_name}!', category='success')
                login_user(user, remember=True)
                return redirect(url_for('views.home'))
            else:
                flash('Incorrect password, try again.', category='error')
        else:
            flash('Email does not exist.', category='error')

    return render_template("login.html", user=current_user)

# Define the route for the logout page
@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))

# Define the route for the sign-up page
@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('firstName')
        password1 = request.form.get('password1')  # Corrected field name
        password2 = request.form.get('password2')  # Corrected field name

        user = User.query.filter_by(email=email).first()

        # Check Sign-up credentials
        if user:
            flash('Email already exists.', category='error')
        elif len(email) < 4:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(first_name) < 2:
            flash('First name must be greater than 1 character.', category='error')
        elif password1 != password2:
            flash('Passwords don\'t match.', category='error')
        elif len(password1) < 7:
            flash('Password must be at least 7 characters.', category='error')
        else:
            new_user = User(email=email, first_name=first_name, password=generate_password_hash(password1, method='sha256'))
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)
            flash('Account Created!!', category='success')
            return redirect(url_for('views.home'))

    return render_template("sign_up.html", user=current_user)
