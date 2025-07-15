from django.db import router
# api/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from django.conf.urls.static import static
from django.conf import settings
from django.urls import path
from script.services.audioService.audio_api import AudioDetectView

from .views.data_views import (
    OperationLogViewSet,
    SubjectViewSet,
    WarningZoneViewSet,
    EventLogViewSet,
    CameraViewSet,
    AlarmLogViewSet,
    UserViewSet,
    RegisterView,
    LoginView,
    DailyReportDataAPI,
    SubmitDailyReportAPI,

)

from .views.feedback_views import FeedbackView
from .views.face_views import (KnownFacesDataAPI,LogEventAPI)

alarm_router = DefaultRouter()
alarm_router.register(r'alarm_logs', AlarmLogViewSet, basename="alarm-log")
router = DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'operation_logs', OperationLogViewSet)
router.register(r'subjects', SubjectViewSet)
router.register(r'warning_zones', WarningZoneViewSet)
router.register(r'cameras', CameraViewSet)
# router.register(r'alarm_logs', AlarmLogViewSet)
router.register(r'event_logs', EventLogViewSet)
# router.register(r'feedback', FeedbackView)

urlpatterns = [
    path('login/', LoginView.as_view(), name='login'),
    path('', include(router.urls)),
    path('register/', RegisterView.as_view(), name='register'),
    path('feedbacks/', FeedbackView.as_view(), name='feedback-list-create'),
    path('', include(alarm_router.urls)),
    path('known-faces/', KnownFacesDataAPI.as_view(), name='ai-known-faces'),
    path('log-event/', LogEventAPI.as_view(), name='ai-log-event'),
    path('daily-report/', DailyReportDataAPI.as_view()),
path('audio_detect/', AudioDetectView.as_view(), name='audio_detect'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

