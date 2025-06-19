"""
Advanced Expense API Routes - Enterprise URL Configuration
Multi-Dimensional Financial Data Endpoints
"""
from django.urls import path
from . import views

# Enterprise Expense Management API Endpoints
urlpatterns = [
    # Core Expense CRUD Operations
    path('', views.AdvancedExpenseListCreate.as_view(), name='expense-list-create'),
    path('<uuid:pk>/', views.ExpenseDetailView.as_view(), name='expense-detail'),

    # Advanced Analytics & Intelligence
    path('analytics/', views.expense_analytics, name='expense-analytics'),
    path('categories/', views.expense_categories, name='expense-categories'),
]