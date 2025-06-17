//
//  ExpenseStore.swift
//  Expense_Tracker
//
//  Created by Sid Kumar on 6/16/25.
//

import Foundation

// MARK: - Expense Store
class ExpenseStore: ObservableObject {
    @Published var expenses: [Expense] = []
    @Published var monthlyIncome: Double?
    @Published var savingsGoal: Double?
    @Published var currentSavings: Double?
    
    var monthlyTotal: Double {
        let now = Date()
        let calendar = Calendar.current
        return expenses.filter { expense in
            calendar.isDate(expense.date, equalTo: now, toGranularity: .month)
        }.reduce(0) { $0 + $1.amount }
    }
    
    var remainingBudget: Double? {
        guard let income = monthlyIncome else { return nil }
        return income - monthlyTotal
    }
    
    var actualSavings: Double? {
        guard let income = monthlyIncome else { return nil }
        return income - monthlyTotal
    }
    
    func addExpense(_ expense: Expense) {
        expenses.insert(expense, at: 0)
    }
    
    func deleteExpense(at offsets: IndexSet) {
        expenses.remove(atOffsets: offsets)
    }
}
