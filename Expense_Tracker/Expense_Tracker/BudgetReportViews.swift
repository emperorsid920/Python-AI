//
//  BudgetReportViews.swift
//  Expense_Tracker
//
//  Created by Sid Kumar on 6/16/25.
//

import SwiftUI

// MARK: - Budget View
struct BudgetView: View {
    @EnvironmentObject var store: ExpenseStore
    @State private var incomeText = ""
    @State private var savingsGoalText = ""
    @State private var currentSavingsText = ""
    
    var body: some View {
        NavigationView {
            Form {
                Section("Current Budget") {
                    HStack {
                        Text("Monthly Income")
                        Spacer()
                        if let income = store.monthlyIncome {
                            Text(String(format: "$%.2f", income))
                                .foregroundColor(.green)
                        } else {
                            Text("Not set")
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    HStack {
                        Text("Current Savings")
                        Spacer()
                        if let savings = store.currentSavings {
                            Text(String(format: "$%.2f", savings))
                                .foregroundColor(.blue)
                        } else {
                            Text("Not set")
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    HStack {
                        Text("Savings Goal")
                        Spacer()
                        if let savingsGoal = store.savingsGoal {
                            Text(String(format: "$%.2f", savingsGoal))
                                .foregroundColor(.purple)
                        } else {
                            Text("Not set")
                                .foregroundColor(.secondary)
                        }
                    }
                }
                
                Section("Update Budget") {
                    TextField("Monthly Income", text: $incomeText)
                        .keyboardType(.decimalPad)
                    TextField("Current Savings", text: $currentSavingsText)
                        .keyboardType(.decimalPad)
                    TextField("Savings Goal", text: $savingsGoalText)
                        .keyboardType(.decimalPad)
                    
                    Button("Update") {
                        updateBudget()
                    }
                }
                
                if store.monthlyIncome != nil {
                    Section("Overview") {
                        BudgetOverview()
                    }
                }
            }
            .navigationTitle("Budget")
            .onAppear {
                loadBudget()
            }
        }
    }
    
    private func loadBudget() {
        if let income = store.monthlyIncome {
            incomeText = String(format: "%.2f", income)
        }
        if let savingsGoal = store.savingsGoal {
            savingsGoalText = String(format: "%.2f", savingsGoal)
        }
        if let currentSavings = store.currentSavings {
            currentSavingsText = String(format: "%.2f", currentSavings)
        }
    }
    
    private func updateBudget() {
        store.monthlyIncome = Double(incomeText)
        store.savingsGoal = Double(savingsGoalText)
        store.currentSavings = Double(currentSavingsText)
    }
}

// MARK: - Budget Overview
struct BudgetOverview: View {
    @EnvironmentObject var store: ExpenseStore
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let income = store.monthlyIncome {
                HStack {
                    Text("Income")
                    Spacer()
                    Text(String(format: "$%.2f", income))
                        .foregroundColor(.green)
                }
                
                HStack {
                    Text("Expenses")
                    Spacer()
                    Text(String(format: "$%.2f", store.monthlyTotal))
                        .foregroundColor(.red)
                }
                
                if let remaining = store.remainingBudget {
                    HStack {
                        Text("Remaining")
                        Spacer()
                        Text(String(format: "$%.2f", remaining))
                            .foregroundColor(remaining >= 0 ? .green : .red)
                    }
                }
                
                if let savings = store.savingsGoal, let actual = store.actualSavings {
                    HStack {
                        Text("Monthly Savings")
                        Spacer()
                        Text(String(format: "$%.2f / $%.2f", actual, savings))
                            .foregroundColor(actual >= savings ? .green : .orange)
                    }
                }
                
                if let currentSavings = store.currentSavings {
                    HStack {
                        Text("Total Savings")
                        Spacer()
                        Text(String(format: "$%.2f", currentSavings))
                            .foregroundColor(.blue)
                    }
                }
                
                if let currentSavings = store.currentSavings, let savingsGoal = store.savingsGoal {
                    HStack {
                        Text("Savings Goal Progress")
                        Spacer()
                        let percentage = (currentSavings / savingsGoal) * 100
                        Text(String(format: "%.1f%%", min(percentage, 100)))
                            .foregroundColor(currentSavings >= savingsGoal ? .green : .orange)
                    }
                }
            }
        }
    }
}

// MARK: - Report View
struct ReportView: View {
    @EnvironmentObject var store: ExpenseStore
    
    var categoryTotals: [ExpenseCategory: Double] {
        var totals: [ExpenseCategory: Double] = [:]
        for expense in store.expenses {
            totals[expense.category, default: 0] += expense.amount
        }
        return totals
    }
    
    var body: some View {
        NavigationView {
            List {
                Section("Summary") {
                    HStack {
                        Text("Total Expenses")
                        Spacer()
                        Text(String(format: "$%.2f", store.expenses.reduce(0) { $0 + $1.amount }))
                    }
                    
                    HStack {
                        Text("This Month")
                        Spacer()
                        Text(String(format: "$%.2f", store.monthlyTotal))
                    }
                    
                    HStack {
                        Text("Transaction Count")
                        Spacer()
                        Text("\(store.expenses.count)")
                    }
                }
                
                if !categoryTotals.isEmpty {
                    Section("Categories") {
                        ForEach(Array(categoryTotals.keys).sorted(by: { categoryTotals[$0]! > categoryTotals[$1]! }), id: \.self) { category in
                            HStack {
                                Image(systemName: category.icon)
                                    .foregroundColor(category.color)
                                Text(category.rawValue)
                                Spacer()
                                Text(String(format: "$%.2f", categoryTotals[category]!))
                            }
                        }
                    }
                }
            }
            .navigationTitle("Report")
        }
    }
}
