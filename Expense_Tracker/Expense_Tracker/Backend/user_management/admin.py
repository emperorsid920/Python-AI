"""
Advanced Administration Interface
Enterprise-Grade User & Financial Data Management
"""
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.utils.html import format_html
from .models import AdvancedUser, ExpenseCategory, ExpenseRecord, BudgetAnalytics


@admin.register(AdvancedUser)
class AdvancedUserAdmin(UserAdmin):
    """
    Enterprise User Administration Panel
    Advanced Financial Profile Management
    """
    list_display = [
        'username', 'email', 'full_name', 'membership_level',
        'monthly_income', 'current_savings', 'account_status', 'created_at'
    ]
    list_filter = [
        'is_premium', 'is_staff', 'is_active', 'created_at'
    ]
    search_fields = ['username', 'email', 'first_name', 'last_name']
    ordering = ['-created_at']

    fieldsets = UserAdmin.fieldsets + (
        ('Financial Profile', {
            'fields': ('monthly_income', 'savings_goal', 'current_savings'),
            'classes': ('collapse',)
        }),
        ('Advanced Profile', {
            'fields': ('phone_number', 'date_of_birth', 'profile_picture', 'is_premium'),
            'classes': ('collapse',)
        }),
    )

    def full_name(self, obj):
        """Display formatted full name"""
        return f"{obj.first_name} {obj.last_name}".strip() or "Not Set"

    full_name.short_description = 'Full Name'

    def membership_level(self, obj):
        """Advanced membership status display"""
        if obj.is_premium:
            return format_html('<span style="color: gold;">🌟 Premium</span>')
        return format_html('<span style="color: blue;">📋 Standard</span>')

    membership_level.short_description = 'Membership'

    def account_status(self, obj):
        """Enhanced account status indicator"""
        if obj.is_active:
            return format_html('<span style="color: green;">✅ Active</span>')
        return format_html('<span style="color: red;">❌ Inactive</span>')

    account_status.short_description = 'Status'


@admin.register(ExpenseCategory)
class ExpenseCategoryAdmin(admin.ModelAdmin):
    """
    Advanced Expense Category Management
    """
    list_display = ['name', 'display_name', 'icon', 'color_preview']
    list_editable = ['icon', 'color_code']

    def display_name(self, obj):
        return obj.get_name_display()

    display_name.short_description = 'Category Name'

    def color_preview(self, obj):
        return format_html(
            '<div style="width: 20px; height: 20px; background-color: {}; border-radius: 50%;"></div>',
            obj.color_code
        )

    color_preview.short_description = 'Color'


@admin.register(ExpenseRecord)
class ExpenseRecordAdmin(admin.ModelAdmin):
    """
    Advanced Expense Record Management System
    Multi-Dimensional Financial Data Analysis
    """
    list_display = [
        'title', 'user', 'formatted_amount', 'category',
        'date', 'location_info', 'created_at'
    ]
    list_filter = ['category', 'date', 'created_at']
    search_fields = ['title', 'user__username', 'notes']
    date_hierarchy = 'date'
    ordering = ['-date', '-created_at']

    fieldsets = [
        ('Expense Details', {
            'fields': ('user', 'title', 'amount', 'category', 'date', 'notes')
        }),
        ('Advanced Analytics', {
            'fields': ('latitude', 'longitude'),
            'classes': ('collapse',)
        }),
    ]

    def formatted_amount(self, obj):
        """Currency formatted amount"""
        return format_html('<strong>${:.2f}</strong>', obj.amount)

    formatted_amount.short_description = 'Amount'
    formatted_amount.admin_order_field = 'amount'

    def location_info(self, obj):
        """Location data display"""
        if obj.latitude and obj.longitude:
            return format_html('📍 {:.4f}, {:.4f}', obj.latitude, obj.longitude)
        return "No Location Data"

    location_info.short_description = 'Location'


@admin.register(BudgetAnalytics)
class BudgetAnalyticsAdmin(admin.ModelAdmin):
    """
    Advanced Budget Analytics Administration
    Real-time Financial Intelligence Dashboard
    """
    list_display = [
        'user', 'current_month_expenses', 'savings_performance',
        'top_category', 'last_updated'
    ]
    list_filter = ['last_updated']
    search_fields = ['user__username']
    readonly_fields = ['last_updated']

    def current_month_expenses(self, obj):
        """Current month expense display"""
        return format_html('<strong>${:.2f}</strong>', obj.total_expenses_current_month)

    current_month_expenses.short_description = 'Current Month'

    def savings_performance(self, obj):
        """Savings rate performance indicator"""
        rate = obj.savings_rate
        if rate >= 20:
            color = 'green'
            icon = '🟢'
        elif rate >= 10:
            color = 'orange'
            icon = '🟡'
        else:
            color = 'red'
            icon = '🔴'

        return format_html(
            '{} <span style="color: {};">{:.1f}%</span>',
            icon, color, rate
        )

    savings_performance.short_description = 'Savings Rate'


# Advanced Admin Site Customization
admin.site.site_header = "Expense Tracker Enterprise Control Panel"
admin.site.site_title = "ET Enterprise Admin"
admin.site.index_title = "Advanced Financial Management System"