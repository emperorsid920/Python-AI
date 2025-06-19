"""
Expense Tracker Backend - Enterprise URL Configuration
Advanced Routing System with Security Protocols
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

# Enterprise API URL Patterns
urlpatterns = [
    # Advanced Administration Portal
    path('admin/', admin.site.urls),

    # Multi-Layer Authentication System
    path('api/auth/login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    # Core Business Logic APIs
    path('api/users/', include('user_management.urls')),
    path('api/expenses/', include('expense_api.urls')),

    # Enterprise API Documentation
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
]

# Development Media Files Handler
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# Advanced Admin Portal Customization
admin.site.site_header = "Expense Tracker Enterprise"
admin.site.site_title = "ET Enterprise Admin"
admin.site.index_title = "Financial Management System"