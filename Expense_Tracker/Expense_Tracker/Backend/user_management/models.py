"""
User Management Module - Advanced Data Models
Enterprise-Grade User Profile & Financial Data Management
"""
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid


class AdvancedUser(AbstractUser):
    """
    Enhanced User Model with Advanced Profile Management
    Multi-Layer Security & Financial Data Integration
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Advanced Profile Fields
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    date_of_birth = models.DateField(blank=True, null=True)
    profile_picture = models.ImageField(upload_to='profiles/', blank=True, null=True)

    # Financial Profile Configuration
    monthly_income = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        blank=True,
        null=True,
        validators=[MinValueValidator(0)]
    )
    savings_goal = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        blank=True,
        null=True,
        validators=[MinValueValidator(0)]
    )
    current_savings = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0.00,
        validators=[MinValueValidator(0)]
    )

    # Advanced Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_premium = models.BooleanField(default=False)

    # Fix reverse accessor conflicts
    groups = models.ManyToManyField(
        'auth.Group',
        verbose_name='groups',
        blank=True,
        help_text='The groups this user belongs to.',
        related_name='advanced_user_set',
        related_query_name='advanced_user',
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        verbose_name='user permissions',
        blank=True,
        help_text='Specific permissions for this user.',
        related_name='advanced_user_set',
        related_query_name='advanced_user',
    )

    class Meta:
        db_table = 'advanced_users'
        verbose_name = 'Advanced User Profile'
        verbose_name_plural = 'Advanced User Profiles'

    def __str__(self):
        return f"{self.username} - Financial Profile"

    @property
    def remaining_budget(self):
        """Calculate remaining monthly budget"""
        if self.monthly_income:
            from django.utils import timezone
            now = timezone.now()
            total_expenses = sum(
                expense.amount for expense in self.expenses.filter(
                    date__month=now.month,
                    date__year=now.year
                )
            )
            return self.monthly_income - total_expenses
        return None


class ExpenseCategory(models.Model):
    """
    Advanced Expense Categorization System
    """
    CATEGORY_CHOICES = [
        ('food', 'Food & Dining'),
        ('transport', 'Transportation'),
        ('entertainment', 'Entertainment'),
        ('shopping', 'Shopping'),
        ('utilities', 'Utilities'),
        ('other', 'Other'),
    ]

    name = models.CharField(max_length=50, choices=CATEGORY_CHOICES, unique=True)
    icon = models.CharField(max_length=50, default='ellipsis.circle.fill')
    color_code = models.CharField(max_length=7, default='#6B7280')

    class Meta:
        verbose_name = 'Expense Category'
        verbose_name_plural = 'Expense Categories'

    def __str__(self):
        return self.get_name_display()


class ExpenseRecord(models.Model):
    """
    Advanced Expense Tracking System
    Multi-Dimensional Financial Data Analysis
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(AdvancedUser, on_delete=models.CASCADE, related_name='expenses')

    # Core Expense Data
    title = models.CharField(max_length=200)
    amount = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0.01)]
    )
    category = models.ForeignKey(ExpenseCategory, on_delete=models.SET_NULL, null=True)
    date = models.DateField()
    notes = models.TextField(blank=True)

    # Advanced Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Geolocation Data (for advanced analytics)
    latitude = models.FloatField(blank=True, null=True)
    longitude = models.FloatField(blank=True, null=True)

    class Meta:
        db_table = 'expense_records'
        verbose_name = 'Expense Record'
        verbose_name_plural = 'Expense Records'
        ordering = ['-date', '-created_at']

    def __str__(self):
        return f"{self.title} - ${self.amount}"


class BudgetAnalytics(models.Model):
    """
    Advanced Budget Analytics & Reporting System
    Real-time Financial Intelligence
    """
    user = models.OneToOneField(AdvancedUser, on_delete=models.CASCADE, related_name='analytics')

    # Monthly Statistics
    total_expenses_current_month = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    total_expenses_last_month = models.DecimalField(max_digits=12, decimal_places=2, default=0)

    # Spending Patterns
    avg_daily_spending = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    top_category = models.CharField(max_length=50, blank=True)

    # Advanced Metrics
    savings_rate = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)]
    )

    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Budget Analytics'
        verbose_name_plural = 'Budget Analytics'

    def __str__(self):
        return f"Analytics for {self.user.username}"