import SwiftUI

struct LoginView: View {
    @State private var username = ""
    @State private var password = ""
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var showHome = false
    @State private var jwtToken: String?

    var body: some View {
        VStack(spacing: 20) {
            Text("Login")
                .font(.largeTitle)
                .bold()

            TextField("Username", text: $username)
                .autocapitalization(.none)
                .textContentType(.username)
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(8)

            SecureField("Password", text: $password)
                .textContentType(.password)
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(8)

            if isLoading {
                ProgressView()
            }

            Button("Login") {
                errorMessage = nil
                guard !username.isEmpty, !password.isEmpty else {
                    errorMessage = "Both fields are required"
                    return
                }
                isLoading = true
                AuthAPI.shared.login(username: username, password: password) { result in
                    DispatchQueue.main.async {
                        isLoading = false
                        switch result {
                        case .success(let loginResponse):
                            jwtToken = loginResponse.access
                            showHome = true // navigate or save token
                        case .failure(let error):
                            errorMessage = error.localizedDescription
                        }
                    }
                }
            }
            .padding()
            .background(Color.blue)
            .foregroundColor(.white)
            .cornerRadius(8)

            if let errorMessage = errorMessage {
                Text(errorMessage)
                    .foregroundColor(.red)
            }

            Spacer()
        }
        .padding()
        .fullScreenCover(isPresented: $showHome) {
            // Replace with your real home view
            Text("Login successful! JWT: \(jwtToken ?? "")")
        }
    }
}
