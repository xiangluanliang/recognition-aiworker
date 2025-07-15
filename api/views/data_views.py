# api/views/data_views.py
import os
from datetime import datetime, time
from datetime import timedelta

from django.contrib.auth import authenticate
from django.contrib.sites import requests
from django.db.models import Count
from django.db.models.functions import TruncDay
from django.utils import timezone
from django.utils.timezone import now
from rest_framework import permissions, viewsets, status
from rest_framework.authtoken.models import Token
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from werkzeug.debug import console

from ..models import (
    User, OperationLog, Subject, WarningZone, Camera, AlarmLog, EventLog, DailyReport
)
from ..serializers import (
    UserSerializer, OperationLogSerializer, SubjectSerializer, WarningZoneSerializer, CameraSerializer,
    AlarmLogSerializer, RegisterSerializer, EventLogSerializer, DailyReportSerializer
)


class IsAdminOrSelf(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        # 管理员全权访问
        if request.user.role_id == 1:
            return True
        # 不是管理员只能访问自己的数据
        return obj.id == request.user.id


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated, IsAdminOrSelf]

    def get_queryset(self):
        user = self.request.user

        # 先检查用户是否已登录认
        if not user.is_authenticated:
            return User.objects.none()

        # 只有当用户登录后，才安全地访问 role_id
        if user.role_id == 1:
            return User.objects.all()
        else:
            return User.objects.filter(pk=user.pk)

    def perform_update(self, serializer):
        user = self.request.user
        # 普通用户不能修改role_id和status
        if user.role_id != 1:
            if 'role_id' in self.request.data or 'status' in self.request.data:
                raise PermissionDenied("无权限修改 role_id 或 status")
        serializer.save()

    def perform_destroy(self, instance):
        user = self.request.user
        # 只有管理员能删除用户
        if user.role_id != 1:
            raise PermissionDenied("无权限删除用户")
        instance.delete()

    @action(detail=False, methods=['get'], url_path='count')
    def count(self, request):
        user_count = User.objects.count()
        return Response({'count': user_count})


class OperationLogViewSet(viewsets.ModelViewSet):
    queryset = OperationLog.objects.all().order_by('-timestamp')
    serializer_class = OperationLogSerializer
    permission_classes = [IsAuthenticated]


class SubjectViewSet(viewsets.ModelViewSet):
    queryset = Subject.objects.all()
    serializer_class = SubjectSerializer
    permission_classes = [IsAuthenticated]

    def create(self, request, *args, **kwargs):
        print("【调试】当前用户是：", request.user)
        return super().create(request, *args, **kwargs)


class WarningZoneViewSet(viewsets.ModelViewSet):
    queryset = WarningZone.objects.all()
    serializer_class = WarningZoneSerializer
    permission_classes = [IsAuthenticated]

    def create(self, request, *args, **kwargs):
        safe_distance = request.data.get('safe_distance')
        safe_time = request.data.get('safe_time')

        print("收到的安全距离:", safe_distance)
        print("收到的安全时间:", safe_time)

        return super().create(request, *args, **kwargs)

class CameraViewSet(viewsets.ModelViewSet):
    queryset = Camera.objects.all()
    serializer_class = CameraSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=["get"])
    def count(self, request):
        total_cameras = self.get_queryset().count()
        return Response({"count": total_cameras})

    @action(detail=False, methods=['get'], url_path='my_cameras')
    def my_cameras(self, request):
        user = request.user
        cameras = Camera.objects.filter(user=user)
        serializer = self.get_serializer(cameras, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'], url_path='flow_info')
    def flow_info(self, request, pk=None):
        try:
            camera = self.get_object()
            # 可加权限判断：只能查看自己的摄像头
            if camera.user != request.user:
                return Response({"detail": "没有权限查看该摄像头"}, status=403)
            return Response({
                "url": camera.url,
                "password": camera.password,
            })
        except Camera.DoesNotExist:
            return Response({"detail": "摄像头不存在"}, status=404)

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context.update({"request": self.request})
        return context


class AlarmLogViewSet(viewsets.ModelViewSet):
    queryset = AlarmLog.objects.select_related('event').all()
    serializer_class = AlarmLogSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=True, methods=['patch'], url_path='update_event_status')
    def update_event_status(self, request, pk=None):
        alarm = self.get_object()
        event = alarm.event

        new_status = request.data.get('status')
        description = request.data.get('description', '')

        if new_status not in [0, 1, 2]:
            return Response({'detail': 'Invalid status'}, status=status.HTTP_400_BAD_REQUEST)

        event.status = new_status
        event.description = description
        event.save()

        return Response({'detail': '事件状态更新成功'})

    def get_queryset(self):
        queryset = AlarmLog.objects.select_related('event').all()
        today_param = self.request.query_params.get("today")

        if today_param == "true":
            today = timezone.now().date()
            queryset = queryset.filter(event__time__date=today)

        return queryset

    @action(detail=True, methods=['get'], url_path='event_detail')
    def event_detail(self, request, pk=None):
        alarm = self.get_object()
        event = alarm.event
        return Response({
            'event_type': event.event_type,
            'event_time': event.time,
            'status': event.status,
            'description': event.description,
            'video_clip_path': event.video_clip_path,
            'image_path': event.image_path,
        })

    @action(detail=False, methods=['get'], url_path='trend')
    def trend(self, request):
        today = now().date()
        start_date = today - timedelta(days=6)  # 最近7天

        alarms = (
            AlarmLog.objects
            .filter(time__date__gte=start_date, time__date__lte=today)
            .annotate(day=TruncDay('time'))
            .values('day')
            .annotate(count=Count('id'))
            .order_by('day')
        )

        date_list = [(start_date + timedelta(days=i)).strftime('%m-%d') for i in range(7)]
        count_dict = {}
        for alarm in alarms:
            day = alarm['day']
            if day:
                count_dict[day.strftime('%m-%d')] = alarm['count']
        count_list = [count_dict.get(date, 0) for date in date_list]
        # print("alarm trend 数据原始结果：", list(alarms))
        return Response({
            "dates": date_list,
            "counts": count_list,
        })


class RegisterView(APIView):
    permission_classes = []  # 注册接口允许匿名访问

    def post(self, request):
        print(request.data)
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"message": "注册成功！"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(APIView):
    permission_classes = []

    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')

        if not username or not password:
            return Response({'message': '用户名和密码不能为空'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return Response({'message': '用户不存在，请先注册'}, status=status.HTTP_401_UNAUTHORIZED)

        user = authenticate(username=username, password=password)
        if user:
            token, created = Token.objects.get_or_create(user=user)
            return Response({
                'token': token.key,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'status': user.status,
                    'created_at': user.created_at,
                    'role': user.role_id.id
                }
            })
        else:
            return Response({'message': '密码错误'}, status=status.HTTP_401_UNAUTHORIZED)


class EventLogViewSet(viewsets.ModelViewSet):
    queryset = EventLog.objects.all().order_by('-time')
    serializer_class = EventLogSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        event = serializer.save()
        print(f"记录新事件：{event.event_type}，来自摄像头：{event.camera_id}")

        if event.event_type in ['fire', 'intrusion', 'conflict', 'face_match']:
            AlarmLog.objects.create(
                title=f"触发告警：{event.get_event_type_display()}",
                event=event,
                time=event.time,
                result=None,
                status=0,
                description=f"检测到 {event.get_event_type_display()}，摄像头ID：{event.camera.id if event.camera else '未知'}"
            )


class DailyReportDataAPI(APIView):
    """
    为本地AI机提供生成日报所需的数据摘要。
    """
    permission_classes = [IsAuthenticated]  # 保护此接口，需要认证

    def get(self, request, *args, **kwargs):
        """
        当接收到GET请求时，从数据库收集数据并返回。
        """
        # 这部分逻辑直接从您的脚本中移入
        today = timezone.localdate()
        start = timezone.make_aware(datetime.combine(today, datetime.min.time()))
        end = timezone.make_aware(datetime.combine(today, datetime.max.time()))

        alarms = AlarmLog.objects.filter(time__range=(start, end))

        summary = {
            '日期': str(today),
            '总事件数': alarms.count(),
            '未处理事件数': alarms.filter(status=0).count(),
            '处理中事件数': alarms.filter(status=1).count(),
            '已处理事件数': alarms.filter(status=2).count(),
            '摄像头总数': Camera.objects.count(),
            '在线摄像头': Camera.objects.filter(is_active=True).count(),
            '各类型事件统计': {},
        }

        type_counts = alarms.values('event__event_type').annotate(count=Count('id'))
        for item in type_counts:
            event_type = item.get('event__event_type')
            if event_type:
                summary['各类型事件统计'][event_type] = item['count']

        return Response(summary, status=status.HTTP_200_OK)


class SubmitDailyReportAPI(APIView):
    """
    接收本地AI机生成的报告文本，并存入数据库。
    """
    permission_classes = [IsAuthenticated]  # 保护此接口，需要认证

    def post(self, request, *args, **kwargs):
        """
        当接收到POST请求时，将报告内容写入数据库。
        """
        report_content = request.data.get('content')
        report_date_str = request.data.get('date')

        if not report_content or not report_date_str:
            return Response(
                {"error": "请求体中必须包含 'content' 和 'date' 字段。"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # 将字符串格式的日期转换回date对象
            report_date = datetime.strptime(report_date_str, '%Y-%m-%d').date()
        except ValueError:
            return Response(
                {"error": "日期格式无效，请使用 'YYYY-MM-DD' 格式。"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 使用 update_or_create 避免重复创建当天的报告
        report, created = DailyReport.objects.update_or_create(
            date=report_date,
            defaults={'content': report_content}
        )

        message = "日报更新成功。" if not created else "日报创建成功。"
        return Response({"status": "success", "message": message}, status=status.HTTP_201_CREATED)
