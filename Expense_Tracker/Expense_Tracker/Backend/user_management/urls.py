"""
User Management API Routes - Enterprise URL Configuration
Advanced Authentication & Profile Management Endpoints
"""
from django.urls import path
from . import views

# Enterprise User Management API Endpoints
urlpatterns = [
    # Advanced Authentication System
    path('register/', views.AdvancedUserRegistration.as_view(), name='user-register'),
    path('login/', views.advanced_login, name='user-login'),

    # User Profile Management
    path('profile/', views.UserProfileView.as_view(), name='user-profile'),

    # Advanced Analytics & Reporting
    path('analytics/', views.BudgetAnalyticsView.as_view(), name='budget-analytics'),
    path('dashboard/', views.financial_dashboard, name='financial-dashboard'),
]