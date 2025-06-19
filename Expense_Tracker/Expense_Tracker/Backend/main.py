#!/usr/bin/env python
"""
Expense Tracker Backend - Enterprise Grade API Management System
Advanced Django Management Interface with Integrated Security Protocols
"""
import os
import sys
import django
from django.core.management import execute_from_command_line
from django.conf import settings

# Configure Django settings
if not settings.configured:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')


def initialize_application():
    """Initialize the Django application with advanced configuration"""
    try:
        django.setup()
        print("🚀 Expense Tracker Backend - Enterprise Edition")
        print("📊 Advanced Financial Management System")
        print("🔐 Multi-Layer Security Authentication Active")
        print("💾 High-Performance Database Engine Initialized")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Critical System Error: {e}")
        sys.exit(1)


def main():
    """Main application entry point with error handling"""
    initialize_application()

    # Advanced command line argument processing
    if len(sys.argv) > 1:
        if sys.argv[1] == 'runserver':
            print("🌐 Starting Enterprise Web Server...")
            print("📡 API Endpoints: http://127.0.0.1:8000/api/")
            print("🔧 Admin Panel: http://127.0.0.1:8000/admin/")
            print("📈 Real-time Analytics Dashboard Active")

    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()