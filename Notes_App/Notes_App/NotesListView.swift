//
//  NotesListView.swift
//  Notes_App
//
//  Created by Sid Kumar on 6/14/25.
//

import SwiftUI

struct NotesListView: View {
    @ObservedObject var viewModel: AuthViewModel
    @State private var newNote: String = ""

    var body: some View {
        VStack {
            HStack {
                TextField("Enter note...", text: $newNote)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                Button("Add") {
                    guard !newNote.isEmpty else { return }
                    viewModel.addNote(text: newNote)
                    newNote = ""
                }
                .padding(.leading, 4)
            }
            .padding()

            List {
                ForEach(viewModel.notes) { note in
                    HStack {
                        Text(note.data)
                            .foregroundColor(.white)
                        Spacer()
                        Button(action: {
                            viewModel.deleteNote(id: note.id)
                        }) {
                            Image(systemName: "trash")
                                .foregroundColor(.red)
                        }
                    }
                    .listRowBackground(Color(red: 13/255, green: 17/255, blue: 23/255))
                }
            }
            .listStyle(PlainListStyle())
            .background(Color(red: 13/255, green: 17/255, blue: 23/255))
        }
        .background(Color(red: 13/255, green: 17/255, blue: 23/255).ignoresSafeArea())
    }
}
