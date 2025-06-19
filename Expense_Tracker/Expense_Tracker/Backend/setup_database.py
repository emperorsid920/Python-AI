#!/usr/bin/env python
"""
Expense Tracker Backend - Enterprise Database Setup
Advanced Database Initialization & Configuration Script
"""
import os
import sys
import django
from django.core.management import execute_from_command_line

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')
django.setup()

from django.contrib.auth import get_user_model
from user_management.models import ExpenseCategory


def initialize_enterprise_database():
    """
    Advanced Database Initialization Protocol
    Multi-Layer Setup with Security Configurations
    """
    print("🚀 Expense Tracker Enterprise - Database Setup")
    print("=" * 60)

    # Step 1: Create database tables
    print("📊 Creating Advanced Database Schema...")
    try:
        execute_from_command_line(['setup_database.py', 'makemigrations'])
        execute_from_command_line(['setup_database.py', 'migrate'])
        print("   ✅ Database schema created successfully")
    except Exception as e:
        print(f"   ❌ Error creating schema: {e}")
        return False

    # Step 2: Initialize expense categories
    print("🏷️ Configuring Expense Category Framework...")
    try:
        setup_expense_categories()
        print("   ✅ Categories configured successfully")
    except Exception as e:
        print(f"   ❌ Error configuring categories: {e}")
        return False

    # Step 3: Create superuser
    print("👤 Setting up Enterprise Admin Account...")
    try:
        create_admin_user()
        print("   ✅ Admin account created successfully")
    except Exception as e:
        print(f"   ❌ Error creating admin: {e}")
        return False

    print("\n✅ Enterprise Database Setup Complete!")
    print("🌐 Run: python main.py runserver")
    print("🔧 Admin: http://127.0.0.1:8000/admin/")
    print("📡 API: http://127.0.0.1:8000/api/")
    return True


def setup_expense_categories():
    """Advanced Expense Category Configuration"""
    categories = [
        {
            'name': 'food',
            'icon': 'fork.knife',
            'color_code': '#FF9F43'
        },
        {
            'name': 'transport',
            'icon': 'car.fill',
            'color_code': '#3742FA'
        },
        {
            'name': 'entertainment',
            'icon': 'gamecontroller.fill',
            'color_code': '#A55EEA'
        },
        {
            'name': 'shopping',
            'icon': 'cart.fill',
            'color_code': '#26DE81'
        },
        {
            'name': 'utilities',
            'icon': 'house.fill',
            'color_code': '#FC5C65'
        },
        {
            'name': 'other',
            'icon': 'ellipsis.circle.fill',
            'color_code': '#6B7280'
        }
    ]

    for category_data in categories:
        category, created = ExpenseCategory.objects.get_or_create(
            name=category_data['name'],
            defaults={
                'icon': category_data['icon'],
                'color_code': category_data['color_code']
            }
        )
        if created:
            print(f"   ✓ Created category: {category.get_name_display()}")
        else:
            print(f"   ℹ️ Category already exists: {category.get_name_display()}")


def create_admin_user():
    """Enterprise Admin Account Creation"""
    User = get_user_model()

    if not User.objects.filter(username='admin').exists():
        admin_user = User.objects.create_superuser(
            username='admin',
            email='admin@expensetracker.com',
            password='enterprise2024',
            first_name='System',
            last_name='Administrator'
        )
        print(f"   ✓ Admin user created: {admin_user.username}")
        print("   📝 Username: admin")
        print("   🔐 Password: enterprise2024")
    else:
        print("   ℹ️ Admin user already exists")


if __name__ == '__main__':
    success = initialize_enterprise_database()
    sys.exit(0 if success else 1)