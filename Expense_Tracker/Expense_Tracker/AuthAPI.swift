import Foundation

struct SignUpRequest: Codable {
    let username: String
    let email: String
    let password: String
}

struct LoginRequest: Codable {
    let username: String
    let password: String
}

struct LoginResponse: Codable {
    let access: String
    let refresh: String
}

enum AuthError: Error, LocalizedError {
    case serverError(String)
    case invalidResponse
    case unknown
    var errorDescription: String? {
        switch self {
        case .serverError(let message): return message
        case .invalidResponse: return "Invalid response from server"
        case .unknown: return "Unknown error"
        }
    }
}

class AuthAPI {
    static let shared = AuthAPI()
    let baseURL = "http://127.0.0.1:8000"

    func signUp(username: String, email: String, password: String, completion: @escaping (Result<Void, Error>) -> Void) {
        guard let url = URL(string: "\(baseURL)/api/users/") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body = SignUpRequest(username: username, email: email, password: password)
        request.httpBody = try? JSONEncoder().encode(body)
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error { completion(.failure(error)); return }
            guard let httpResponse = response as? HTTPURLResponse else {
                completion(.failure(AuthError.invalidResponse)); return
            }
            if (200...299).contains(httpResponse.statusCode) {
                completion(.success(()))
            } else {
                // Try to decode error message from backend
                if let data = data, let errorMsg = String(data: data, encoding: .utf8) {
                    completion(.failure(AuthError.serverError(errorMsg)))
                } else {
                    completion(.failure(AuthError.unknown))
                }
            }
        }.resume()
    }

    func login(username: String, password: String, completion: @escaping (Result<LoginResponse, Error>) -> Void) {
        guard let url = URL(string: "\(baseURL)/api/auth/login/") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body = LoginRequest(username: username, password: password)
        request.httpBody = try? JSONEncoder().encode(body)
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error { completion(.failure(error)); return }
            guard let data = data else {
                completion(.failure(AuthError.invalidResponse)); return
            }
            do {
                let result = try JSONDecoder().decode(LoginResponse.self, from: data)
                completion(.success(result))
            } catch {
                if let errorMsg = String(data: data, encoding: .utf8) {
                    completion(.failure(AuthError.serverError(errorMsg)))
                } else {
                    completion(.failure(AuthError.unknown))
                }
            }
        }.resume()
    }
}
