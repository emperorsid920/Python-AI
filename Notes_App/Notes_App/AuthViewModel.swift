//
//  AuthViewModel.swift
//  Notes_App
//
//  Created by Sid Kumar on 6/14/25.
//
import Foundation

class AuthViewModel: ObservableObject {
    @Published var email = ""
    @Published var password = ""
    @Published var firstName = ""
    @Published var isAuthenticated = false
    @Published var errorMessage: String?
    @Published var isRegistering = false
    @Published var notes: [Note] = []

    private var token: String?

    let backendURL = "http://127.0.0.1:5000/api" // Change to your IP for real device

    func login(completion: @escaping () -> Void) {
        guard let url = URL(string: "\(backendURL)/login") else { return }
        let params = ["email": email, "password": password]
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: params)

        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let data = data,
                   let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let token = dict["access_token"] as? String {
                    self?.token = token
                    self?.isAuthenticated = true
                    self?.errorMessage = nil
                    self?.getNotes()
                } else {
                    self?.errorMessage = "Login failed"
                }
                completion()
            }
        }.resume()
    }

    func register(completion: @escaping () -> Void) {
        guard let url = URL(string: "\(backendURL)/register") else { return }
        let params = ["email": email, "first_name": firstName, "password": password]
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: params)

        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let data = data,
                   let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let token = dict["access_token"] as? String {
                    self?.token = token
                    self?.isAuthenticated = true
                    self?.errorMessage = nil
                    self?.getNotes()
                } else {
                    self?.errorMessage = "Registration failed"
                }
                completion()
            }
        }.resume()
    }

    func getNotes() {
        guard let token = token,
              let url = URL(string: "\(backendURL)/notes") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let data = data,
                   let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let notesArray = dict["notes"] as? [[String: Any]] {
                    self?.notes = notesArray.compactMap { Note(json: $0) }
                }
            }
        }.resume()
    }

    func addNote(text: String, completion: (() -> Void)? = nil) {
        guard let token = token,
              let url = URL(string: "\(backendURL)/notes") else { return }
        let params = ["data": text]
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.httpBody = try? JSONSerialization.data(withJSONObject: params)

        // Optimistically append the note with a temporary ID
        let tempID = (notes.max(by: { $0.id < $1.id })?.id ?? 0) + 1
        let newNote = Note(id: tempID, data: text)
        notes.append(newNote)

        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                self?.getNotes() // Refresh from backend (to get real ID)
                completion?()
            }
        }.resume()
    }

    func deleteNote(id: Int) {
        guard let token = token,
              let url = URL(string: "\(backendURL)/notes/\(id)") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

        // Optimistically remove from UI
        notes.removeAll { $0.id == id }

        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                self?.getNotes()
            }
        }.resume()
    }

    func logout() {
        token = nil
        email = ""
        password = ""
        firstName = ""
        isAuthenticated = false
        notes = []
        errorMessage = nil
        isRegistering = false
    }
}
