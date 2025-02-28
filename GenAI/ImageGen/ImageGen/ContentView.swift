//
//  ContentView.swift
//  ImageGen
//
//  Created by Sid Kumar on 2/24/24.
//

import SwiftUI

struct ContentView: View {
    @State private var userInput: String = ""

    var body: some View {
        VStack {
            TextField("Enter text", text: $userInput)
                .padding()
                .textFieldStyle(RoundedBorderTextFieldStyle())
            
            Button(action: {
                // Handle submit button action here
                print("Submitted: \(userInput)")
                // You can perform further actions here, such as submitting the input data
            }) {
                Text("Submit")
                    .padding()
                    .foregroundColor(.white)
                    .background(Color.accentColor)
                    .cornerRadius(8)
            }
            .padding()
            
            Spacer() // Optional spacer to push views to the top of the screen
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

