//
//  ContentView.swift
//  Notes_App
//
//  Created by Sid Kumar on 6/13/25.
//
import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = AuthViewModel()
    @State private var showNotes = false

    var body: some View {
        NavigationView {
            ZStack {
                Color(red: 13/255, green: 17/255, blue: 23/255).ignoresSafeArea()
                VStack(spacing: 20) {
                    if !showNotes && !viewModel.isAuthenticated {
                        VStack(spacing: 10) {
                            TextField("Email", text: $viewModel.email)
                                .autocapitalization(.none)
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                            SecureField("Password", text: $viewModel.password)
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                            if viewModel.isRegistering {
                                TextField("First Name", text: $viewModel.firstName)
                                    .textFieldStyle(RoundedBorderTextFieldStyle())
                            }
                            Button(viewModel.isRegistering ? "Register" : "Login") {
                                if viewModel.isRegistering {
                                    viewModel.register {
                                        showNotes = viewModel.isAuthenticated
                                    }
                                } else {
                                    viewModel.login {
                                        showNotes = viewModel.isAuthenticated
                                    }
                                }
                            }
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(8)

                            Button(viewModel.isRegistering ? "Already have an account? Login" : "No account? Register") {
                                viewModel.isRegistering.toggle()
                            }
                            .font(.caption)
                            .padding(.top, 5)
                        }
                    } else {
                        NotesListView(viewModel: viewModel)
                        Button("Logout") {
                            viewModel.logout()
                            showNotes = false
                        }
                        .padding()
                        .background(Color.red)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                        .padding(.top, 12)
                    }
                    if let error = viewModel.errorMessage {
                        Text(error)
                            .foregroundColor(.red)
                    }
                }
                .padding()
            }
            .navigationTitle("Notes App")
        }
    }
}
