import SwiftUI

struct SignUpView: View {
    @State private var username = ""
    @State private var email = ""
    @State private var password = ""
    @State private var confirmPassword = ""
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var showSuccess = false

    var body: some View {
        VStack(spacing: 20) {
            Text("Sign Up")
                .font(.largeTitle)
                .bold()

            TextField("Username", text: $username)
                .autocapitalization(.none)
                .textContentType(.username)
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(8)

            TextField("Email", text: $email)
                .autocapitalization(.none)
                .keyboardType(.emailAddress)
                .textContentType(.emailAddress)
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(8)

            SecureField("Password", text: $password)
                .textContentType(.newPassword)
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(8)

            SecureField("Confirm Password", text: $confirmPassword)
                .textContentType(.newPassword)
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(8)

            if isLoading {
                ProgressView()
            }

            Button("Create Account") {
                errorMessage = nil
                guard !username.isEmpty, !email.isEmpty, !password.isEmpty else {
                    errorMessage = "All fields are required"
                    return
                }
                guard password == confirmPassword else {
                    errorMessage = "Passwords do not match"
                    return
                }
                isLoading = true
                AuthAPI.shared.signUp(username: username, email: email, password: password) { result in
                    DispatchQueue.main.async {
                        isLoading = false
                        switch result {
                        case .success():
                            showSuccess = true
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
        .alert(isPresented: $showSuccess) {
            Alert(
                title: Text("Success"),
                message: Text("Account created. Please log in."),
                dismissButton: .default(Text("OK"))
            )
        }
    }
}
