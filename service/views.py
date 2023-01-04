import json
import os.path

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage

import os
from django.http import FileResponse
# Create your views here.

# formdata 관련: https://uni2237.tistory.com/10
# 업로드 다운로드 기능: https://triple-f-coding.tistory.com/5
# 파일 다운로드 기능 https://wonpaper.tistory.com/466

# 다운 로드: https://hyundy.tistory.com/11

# 메인화면
def mainView(request):
    return render(request, 'service/index.html')

# 내가 마신 칵테일과 비슷한 칵테일은?
def cluster(request):
    return render(request, 'service/cluster.html')

def clusterAjax(request):
    print("cluasterAjax 호출")
    context = {
        'private' : 5
    }
    return JsonResponse(context)


# 지금 만들 수 있는 칵테일은?
def ingredient(request):
    return render(request, 'service/ingredient.html')

# 이 칵테일의 이름은?
def image(request):
    return render(request, 'service/image.html')

@csrf_exempt
def imageAjax(request):
    print("imageAjax 호출")
    context = {
        'private' : 15
    }


    print(request.FILES['uploadFile'])
    img = request.FILES['uploadFile']
    fs = FileSystemStorage(location='media/', base_url='media/')
    fs.save(img.name, img)

    return JsonResponse(context)


def downloadFile(request):
    file_path = os.path.abspath('media/')
    file_name = os.path.basename("media/오류.png")
    fs = FileSystemStorage(file_path)

    response = FileResponse(fs.open(file_name, 'rb'))

    response['Content-Disposition'] = 'attachment; filename="오류.png"'

    return response



# 칵테일 사진 화풍변경

def changeImage(request):
    return render(request, 'service/changeImage.html')

# 사용 기술
def tech(request):
    return render(request, 'service/tech.html')

# 주변 칵테일 바의 위치는?
def barLocation(request):
    return render(request, 'service/barLocation.html')





