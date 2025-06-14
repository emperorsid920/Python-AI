//
//  Note.swift
//  Notes_App
//
//  Created by Sid Kumar on 6/14/25.
//

import Foundation

struct Note: Identifiable {
    let id: Int
    let data: String

    init(id: Int, data: String) {
        self.id = id
        self.data = data
    }

    init?(json: [String: Any]) {
        guard let id = json["id"] as? Int,
              let data = json["data"] as? String else {
            return nil
        }
        self.id = id
        self.data = data
    }
}
