import json
import os.path

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from service.models import Uploadimage, Uploadimagelog
from datetime import datetime
from pytz import timezone

import os
from django.http import FileResponse
# Create your views here.

# formdata 관련: https://uni2237.tistory.com/10
# 업로드 다운로드 기능: https://triple-f-coding.tistory.com/5
# 파일 다운로드 기능 https://wonpaper.tistory.com/466

# 다운 로드: https://hyundy.tistory.com/11


"""
service/index에서
True를 리턴할 경우에, '로그아웃' 버튼이 생기기 위해서.
False를 리턴할 경우에, 현재 접속 아이디를 html에서 표현 가능하도록.

"""
def logInOut(request):
    context = {}

    login_session = request.session.get('email', '')

    if login_session == '':
        context['login_session'] = False

    else:
        # 값이 있으면,
        context['login_session'] = True
        context['email'] = request.session['email']

        # print(login_session['email'])
        # context['email'] = login_session['email']

    return context


"""
service/index에서
리턴 False이면, index로 리다이렉트가 되도록.

"""

def logInCheck(request):
    login_session = request.session.get('email', '')

    if login_session == '':
        # print("session에 저장된 아이디가 없습니다.")
        return False

    else:
        return True



# 메인화면
def mainView(request):

    if logInCheck(request):
        context = logInOut(request)
        print(request.session['email'])

        return render(request, 'service/index.html', context)
    else:
        # print('비정상적인 접근: session에 id가 없다.')
        return redirect('/')


# 내가 마신 칵테일과 비슷한 칵테일은?
def cluster(request):

    if logInCheck(request):
        print("현재 로그인 이메일 " + request.session['email'])
        return render(request, 'service/cluster.html')
    else:
        print('비정상적인 접근: session에 id가 없다.')
        return redirect('/')




def clusterAjax(request):

    print("cluasterAjax 호출")
    context = {
        'private' : 5
    }
    return JsonResponse(context)


# 지금 만들 수 있는 칵테일은?
def ingredient(request):

    if logInCheck(request):
        print("현재 로그인 이메일 " + request.session['email'])
        return render(request, 'service/ingredient.html')
    else:
        print('비정상적인 접근: session에 id가 없다.')
        return redirect('/')

    # return render(request, 'service/ingredient.html')

# 이 칵테일의 이름은?
def image(request):

    if logInCheck(request):
        print("현재 로그인 이메일 " + request.session['email'])
        return render(request, 'service/image.html')
    else:
        print('비정상적인 접근: session에 id가 없다.')
        return redirect('/')

    # return render(request, 'service/image.html')

@csrf_exempt
def imageAjax(request):
    context = {
        'private' : 15
    }

    img = request.FILES.get('uploadFile')

    imageName = img.name
    trimImageName = imageName.replace(" ", "")


    email = request.session['email']
    day = str(datetime.now(timezone('Asia/Seoul')).date())

    dotPosition = trimImageName.find('.')

    hadoopFileName = trimImageName + day

    hadoopFileName = trimImageName[0:dotPosition] +":"+ day + trimImageName[dotPosition: len(trimImageName)]


    uploadImage = Uploadimage(email=email, filename=trimImageName, register_date= day, hadoopfilename= hadoopFileName)
    uploadImageLog = Uploadimagelog(email=email, filename=trimImageName, register_date=day, hadoopfilename=hadoopFileName)

    uploadImage.save()
    uploadImageLog.save()
    """
    로컬 테스트용: 서버 테스트용을 구별해서 주석을 제거할것.
    """

    # TODO 서버용 경로와 로컬 테스트 경로를 구분해서 업로드해야 합니다. 깃헙 PUSH시, 서버 경로로.


    ## 서버용 경로
    fs = FileSystemStorage(location='/home/jwjinn/attachement/images', base_url='/home/jwjinn/attachement/images')

    ## 로컬 경로(주우진)
    # fs = FileSystemStorage(location='/home/joo/images', base_url='/home/joo/images')

    fs.save(hadoopFileName, img)

    return JsonResponse(context)


# TODO: 다운이 되도록 수정을 할 것.
def downloadFile(request):
    file_path = os.path.abspath('media/')
    file_name = os.path.basename("media/오류.png")
    fs = FileSystemStorage(file_path)

    response = FileResponse(fs.open(file_name, 'rb'))

    response['Content-Disposition'] = 'attachment; filename="오류.png"'

    return response



# 칵테일 사진 화풍변경

def changeImage(request):

    if logInCheck(request):
        print("현재 로그인 이메일 " + request.session['email'])
        return render(request, 'service/changeImage.html')
    else:
        print('비정상적인 접근: session에 id가 없다.')
        return redirect('/')

    # return render(request, 'service/changeImage.html')

# 사용 기술
def tech(request):

    if logInCheck(request):
        print("현재 로그인 이메일 " + request.session['email'])
        return render(request, 'service/tech.html')
    else:
        print('비정상적인 접근: session에 id가 없다.')
        return redirect('/')

    # return render(request, 'service/tech.html')

# 주변 칵테일 바의 위치는?
def barLocation(request):

    if logInCheck(request):
        print("현재 로그인 이메일 " + request.session['email'])
        return render(request, 'service/barLocation.html')
    else:
        print('비정상적인 접근: session에 id가 없다.')
        return redirect('/')

    # return render(request, 'service/barLocation.html')





