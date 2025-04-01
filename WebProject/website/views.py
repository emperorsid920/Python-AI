# Library for rendering a template
from flask import Blueprint, render_template, request, flash, jsonify

from flask_login import login_required, current_user
from .models import Note
from . import db
import json

# This file is the blueprint of our application -> URLs are defined here
# Stores the standard routes for our website -> where users can go to

views = Blueprint('views', __name__)  # defining views blueprint

# Remove these imports from the top of the file
# from . import db
# from .models import Note

# Defining a route or a URL
@views.route('/', methods=['GET', 'POST'])  # defining the homepage('/')
@login_required
def home():  # Function will run whenever a user goes to the main page
    # Move these imports inside the function where they are used
    from .models import Note
    from . import db

    if request.method == 'POST':
        note = request.form.get('note')

        if len(note) < 1:
            flash('Note is too short!!', category='error')
        else:
            new_note = Note(data=note, user_id=current_user.id)
            db.session.add(new_note)
            db.session.commit()
            flash('Note added!!', category='success')

    return render_template("home.html", user=current_user)  # returning the HTML file for rendering

@views.route('/delete-note', methods=['POST'])
def delete_note():
    note = json.loads(request.data)                    #this function expects a JSON from the INDEX.js file
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()

    return jsonify({}  )
