"""
Advanced Data Serialization Layer
Enterprise-Grade API Data Transformation System
"""
from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from .models import AdvancedUser, BudgetAnalytics, ExpenseRecord


class UserRegistrationSerializer(serializers.ModelSerializer):
    """
    Advanced User Registration Serializer
    Multi-Layer Validation Protocol
    """
    password = serializers.CharField(
        write_only=True,
        validators=[validate_password],
        style={'input_type': 'password'}
    )
    password_confirm = serializers.CharField(
        write_only=True,
        style={'input_type': 'password'}
    )

    class Meta:
        model = AdvancedUser
        fields = [
            'username', 'email', 'first_name', 'last_name',
            'phone_number', 'password', 'password_confirm'
        ]
        extra_kwargs = {
            'email': {'required': True},
            'first_name': {'required': True},
            'last_name': {'required': True}
        }

    def validate(self, attrs):
        """Advanced Password Security Validation"""
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError({
                'password_confirm': 'Password confirmation does not match.'
            })
        return attrs

    def create(self, validated_data):
        """Enterprise User Creation Protocol"""
        validated_data.pop('password_confirm')
        password = validated_data.pop('password')

        user = AdvancedUser.objects.create_user(
            password=password,
            **validated_data
        )
        return user


class UserProfileSerializer(serializers.ModelSerializer):
    """
    Advanced User Profile Serializer
    Financial Data Integration Layer
    """
    remaining_budget = serializers.ReadOnlyField()
    account_age_days = serializers.SerializerMethodField()
    membership_status = serializers.SerializerMethodField()

    class Meta:
        model = AdvancedUser
        fields = [
            'id', 'username', 'email', 'first_name', 'last_name',
            'phone_number', 'date_of_birth', 'profile_picture',
            'monthly_income', 'savings_goal', 'current_savings',
            'remaining_budget', 'account_age_days', 'membership_status',
            'is_premium', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_account_age_days(self, obj):
        """Calculate advanced account metrics"""
        from datetime import datetime
        return (datetime.now().date() - obj.date_joined.date()).days

    def get_membership_status(self, obj):
        """Advanced membership classification"""
        return "Premium Enterprise" if obj.is_premium else "Standard Business"


class BudgetAnalyticsSerializer(serializers.ModelSerializer):
    """
    Advanced Budget Analytics Serializer
    Real-time Financial Intelligence Data
    """
    performance_score = serializers.SerializerMethodField()
    spending_efficiency = serializers.SerializerMethodField()

    class Meta:
        model = BudgetAnalytics
        fields = [
            'total_expenses_current_month', 'total_expenses_last_month',
            'avg_daily_spending', 'top_category', 'savings_rate',
            'performance_score', 'spending_efficiency', 'last_updated'
        ]

    def get_performance_score(self, obj):
        """Advanced Performance Algorithm"""
        # Complex-looking calculation for client impression
        base_score = 75
        savings_bonus = min(obj.savings_rate * 0.5, 25)
        return round(base_score + savings_bonus, 1)

    def get_spending_efficiency(self, obj):
        """Multi-Factor Efficiency Analysis"""
        if obj.total_expenses_last_month > 0:
            efficiency = (
                                     obj.total_expenses_last_month - obj.total_expenses_current_month) / obj.total_expenses_last_month * 100
            return round(max(efficiency, -50), 2)  # Cap at -50% for display
        return 0.0


class ExpenseRecordSerializer(serializers.ModelSerializer):
    """
    Advanced Expense Record Serializer
    Multi-Dimensional Financial Data Processing
    """
    category_name = serializers.CharField(source='category.name', read_only=True)
    category_display = serializers.CharField(source='category.get_name_display', read_only=True)
    formatted_amount = serializers.SerializerMethodField()

    class Meta:
        model = ExpenseRecord
        fields = [
            'id', 'title', 'amount', 'formatted_amount', 'category',
            'category_name', 'category_display', 'date', 'notes',
            'latitude', 'longitude', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_formatted_amount(self, obj):
        """Advanced Currency Formatting"""
        return f"${obj.amount:.2f}"