from flask import Blueprint, request, jsonify
from .models import db, User, Note
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import (
    create_access_token, jwt_required, get_jwt_identity
)
from datetime import timedelta

api_bp = Blueprint('api', __name__)

# User Registration
@api_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    first_name = data.get('first_name')
    password = data.get('password')

    if not email or not first_name or not password:
        return jsonify({'error': 'Missing fields'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409

    hashed_pw = generate_password_hash(password, method='sha256')
    user = User(email=email, first_name=first_name, password=hashed_pw)
    db.session.add(user)
    db.session.commit()

    return jsonify({'message': 'User registered'}), 201

# User Login
@api_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password, password):
        return jsonify({'error': 'Invalid credentials'}), 401

    access_token = create_access_token(
        identity=user.id, expires_delta=timedelta(days=7)
    )
    return jsonify({
        'access_token': access_token,
        'user': {'id': user.id, 'email': user.email, 'first_name': user.first_name}
    }), 200

# Get all notes for current user
@api_bp.route('/notes', methods=['GET'])
@jwt_required()
def get_notes():
    user_id = get_jwt_identity()
    notes = Note.query.filter_by(user_id=user_id).order_by(Note.date.desc()).all()
    notes_data = [
        {'id': n.id, 'data': n.data, 'date': n.date.isoformat()}
        for n in notes
    ]
    return jsonify(notes=notes_data), 200

# Add a note
@api_bp.route('/notes', methods=['POST'])
@jwt_required()
def add_note():
    user_id = get_jwt_identity()
    data = request.get_json()
    note_data = data.get('data')

    if not note_data or len(note_data) < 1:
        return jsonify({'error': 'Note is too short'}), 400

    note = Note(data=note_data, user_id=user_id)
    db.session.add(note)
    db.session.commit()

    return jsonify({
        'message': 'Note added',
        'note': {'id': note.id, 'data': note.data, 'date': note.date.isoformat()}
    }), 201

# Delete a note
@api_bp.route('/notes/<int:note_id>', methods=['DELETE'])
@jwt_required()
def delete_note(note_id):
    user_id = get_jwt_identity()
    note = Note.query.get(note_id)
    if not note or note.user_id != user_id:
        return jsonify({'error': 'Note not found'}), 404

    db.session.delete(note)
    db.session.commit()
    return jsonify({'message': 'Note deleted'}), 200

# Optionally: Get user profile
@api_bp.route('/profile', methods=['GET'])
@jwt_required()
def profile():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({'id': user.id, 'email': user.email, 'first_name': user.first_name}), 200