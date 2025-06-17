//
//  ExpenseViews.swift
//  Expense_Tracker
//
//  Created by Sid Kumar on 6/16/25.
//

import SwiftUI

// MARK: - Expense List View
struct ExpenseListView: View {
    @EnvironmentObject var store: ExpenseStore
    
    var body: some View {
        NavigationView {
            VStack {
                // Header
                VStack {
                    Text("Monthly Total")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    
                    Text(String(format: "$%.2f", store.monthlyTotal))
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    if let remaining = store.remainingBudget {
                        Text("Remaining: \(String(format: "$%.2f", remaining))")
                            .foregroundColor(remaining >= 0 ? .green : .red)
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                
                // List
                if store.expenses.isEmpty {
                    Spacer()
                    Text("No expenses yet")
                        .font(.title2)
                        .foregroundColor(.secondary)
                    Spacer()
                } else {
                    List {
                        ForEach(store.expenses) { expense in
                            ExpenseRow(expense: expense)
                        }
                        .onDelete(perform: store.deleteExpense)
                    }
                }
            }
            .navigationTitle("Expenses")
        }
    }
}

// MARK: - Expense Row
struct ExpenseRow: View {
    let expense: Expense
    
    var body: some View {
        HStack {
            Image(systemName: expense.category.icon)
                .foregroundColor(expense.category.color)
                .frame(width: 30)
            
            VStack(alignment: .leading) {
                Text(expense.title)
                    .font(.headline)
                Text(expense.category.rawValue)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Text(expense.formattedAmount)
                .font(.headline)
                .fontWeight(.semibold)
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Add Expense View
struct AddExpenseView: View {
    @EnvironmentObject var store: ExpenseStore
    @State private var title = ""
    @State private var amount = ""
    @State private var category = ExpenseCategory.other
    @State private var date = Date()
    @State private var notes = ""
    @State private var showAlert = false
    
    var body: some View {
        NavigationView {
            Form {
                Section("Details") {
                    TextField("Title", text: $title)
                    TextField("Amount", text: $amount)
                        .keyboardType(.decimalPad)
                    
                    Picker("Category", selection: $category) {
                        ForEach(ExpenseCategory.allCases, id: \.self) { cat in
                            HStack {
                                Image(systemName: cat.icon)
                                    .foregroundColor(cat.color)
                                Text(cat.rawValue)
                            }
                        }
                    }
                    
                    DatePicker("Date", selection: $date, displayedComponents: .date)
                }
                
                Section("Notes") {
                    TextField("Optional notes", text: $notes)
                }
                
                Button("Add Expense") {
                    addExpense()
                }
                .disabled(title.isEmpty || amount.isEmpty)
            }
            .navigationTitle("Add Expense")
            .alert("Added!", isPresented: $showAlert) {
                Button("OK") { }
            }
        }
    }
    
    private func addExpense() {
        guard let amountValue = Double(amount), amountValue > 0 else { return }
        
        let expense = Expense(
            title: title,
            amount: amountValue,
            category: category,
            date: date,
            notes: notes
        )
        
        store.addExpense(expense)
        
        // Reset form
        title = ""
        amount = ""
        category = .other
        date = Date()
        notes = ""
        
        showAlert = true
    }
}
