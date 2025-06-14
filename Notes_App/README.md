# Notes App

A simple full-stack notes-taking application with a **Flask backend** and a **SwiftUI iOS frontend**.  
Users can register, log in, add, view, and delete notes securely.

---

## Features

- User registration & login (JWT authentication)
- Add, view, and delete personal notes
- Persistent storage with SQLite (Flask backend)
- Clean, dark-themed SwiftUI interface
- Logout support
- Optimistic UI updates for notes

---

## Project Structure

```
Notes_App/
├── Backend/           # Flask backend (API, models, etc.)
│   ├── __init__.py
│   ├── main.py        # Entry point for Flask app
│   ├── ...            # Other backend files
├── Notes_App/         # Xcode project directory (iOS app)
│   ├── Notes_AppApp.swift
│   ├── ContentView.swift
│   ├── AuthViewModel.swift
│   ├── NotesListView.swift
│   ├── Note.swift
│   └── ...            # Other Swift files
├── Notes_App.xcodeproj
└── README.md
```

---

## Getting Started

### 1. Backend Setup (Flask)

1. **Install dependencies**  
   Navigate to the backend directory and install requirements:
   ```bash
   cd Notes_App
   pip install flask flask_sqlalchemy flask_cors flask_jwt_extended
   ```

2. **Run the backend server**
   ```bash
   python main.py
   ```
   - By default, the API runs at `http://127.0.0.1:5000`
   - Endpoints: `/api/register`, `/api/login`, `/api/notes` (GET/POST), `/api/notes/<id>` (DELETE)

---

### 2. iOS App Setup (SwiftUI)

1. **Open in Xcode**
   - Double-click `Notes_App.xcodeproj` to open the project in Xcode.

2. **Configure Backend URL (if needed)**
   - In `AuthViewModel.swift`, set:
     ```swift
     let backendURL = "http://127.0.0.1:5000/api"
     ```
     - If testing on a real device, replace `127.0.0.1` with your computer’s LAN IP.

3. **Build and Run**
   - Select a simulator (e.g., iPhone 15)
   - Press `Cmd+R` or click the ▶️ button

---

## Screenshots

| Login/Register | Notes List & Add | Dark Theme |
|:--------------:|:----------------:|:----------:|
| ![Login](screenshots/login.png) | ![Notes](screenshots/notes.png) | ![Dark](screenshots/dark.png) |

---

## API Endpoints

- `POST /api/register` — Registers a new user (`email`, `password`, `first_name`)
- `POST /api/login` — Returns JWT token for authentication (`email`, `password`)
- `GET /api/notes` — Gets notes for logged-in user (JWT in Authorization header)
- `POST /api/notes` — Adds a note (`data`)
- `DELETE /api/notes/<id>` — Deletes note by ID

---

## Customization

- To change the theme color, edit the background color in `ContentView.swift` and `NotesListView.swift`.
- Add more features (note editing, user profile, etc.) by extending the backend and frontend code.

---

## License

MIT License.  
Feel free to use and modify this project.

---

## Acknowledgments

- [Flask](https://palletsprojects.com/p/flask/)
- [SwiftUI](https://developer.apple.com/xcode/swiftui/)
- Inspired by GitHub’s dark theme.
