//
//  Models.swift
//  Expense_Tracker
//
//  Created by Sid Kumar on 6/16/25.
//

import SwiftUI
import Foundation

// MARK: - Expense Model
struct Expense: Identifiable, Codable {
    let id = UUID()
    var title: String
    var amount: Double
    var category: ExpenseCategory
    var date: Date
    var notes: String
    
    var formattedAmount: String {
        String(format: "$%.2f", amount)
    }
}

// MARK: - Expense Category
enum ExpenseCategory: String, CaseIterable, Codable {
    case food = "Food"
    case transport = "Transport"
    case entertainment = "Entertainment"
    case shopping = "Shopping"
    case utilities = "Utilities"
    case other = "Other"
    
    var icon: String {
        switch self {
        case .food: return "fork.knife"
        case .transport: return "car.fill"
        case .entertainment: return "gamecontroller.fill"
        case .shopping: return "cart.fill"
        case .utilities: return "house.fill"
        case .other: return "ellipsis.circle.fill"
        }
    }
    
    var color: Color {
        switch self {
        case .food: return .orange
        case .transport: return .blue
        case .entertainment: return .purple
        case .shopping: return .green
        case .utilities: return .red
        case .other: return .gray
        }
    }
}
