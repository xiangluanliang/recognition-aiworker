from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
import tempfile
from script.services.audioService.video_audio_handler import analyze_video_audio

class AudioDetectView(APIView):
    def post(self, request):
        print("[AudioDetectView] 收到请求")

        if 'file' not in request.data:
            print("[AudioDetectView] 未上传文件")
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        file_obj = request.data['file']

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            for chunk in file_obj.chunks():
                temp_file.write(chunk)
            temp_video_path = temp_file.name

        print(f"[AudioDetectView] 临时保存路径: {temp_video_path}")

        try:
            results = analyze_video_audio(temp_video_path)
            print(f"[AudioDetectView] 返回结果: {results}")
            os.unlink(temp_video_path)
            return Response({"results": results})
        except Exception as e:
            print(f"[AudioDetectView] 异常: {str(e)}")
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
