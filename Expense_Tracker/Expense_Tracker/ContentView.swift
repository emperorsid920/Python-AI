//
//  ContentView.swift
//  Expense_Tracker
//
//  Created by Sid Kumar on 6/16/25.
//

import SwiftUI

struct ContentView: View {
    @State private var showLogin = true

    var body: some View {
        NavigationView {
            VStack {
                if showLogin {
                    LoginView()
                    Button("Don't have an account? Sign Up") {
                        showLogin = false
                    }
                    .padding(.top)
                } else {
                    SignUpView()
                    Button("Already have an account? Login") {
                        showLogin = true
                    }
                    .padding(.top)
                }
            }
            .padding()
        }
    }
}
