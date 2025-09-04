from django.urls import path
from .views import UploadDocView, ChatView

urlpatterns = [
    path('api/upload/', UploadDocView.as_view(), name="upload_doc"),
    path('api/ask/', ChatView.as_view(), name="chat_query")
]
