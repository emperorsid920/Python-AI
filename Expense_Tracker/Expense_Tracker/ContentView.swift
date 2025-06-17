//
//  ContentView.swift
//  Expense_Tracker
//
//  Created by Sid Kumar on 6/16/25.
//

import SwiftUI

// MARK: - Main Content View
struct ContentView: View {
    @StateObject private var store = ExpenseStore()
    
    var body: some View {
        TabView {
            ExpenseListView()
                .tabItem {
                    Image(systemName: "list.bullet")
                    Text("Expenses")
                }
            
            AddExpenseView()
                .tabItem {
                    Image(systemName: "plus")
                    Text("Add")
                }
            
            BudgetView()
                .tabItem {
                    Image(systemName: "dollarsign.circle")
                    Text("Budget")
                }
            
            ReportView()
                .tabItem {
                    Image(systemName: "chart.bar")
                    Text("Report")
                }
        }
        .environmentObject(store)
    }
}
