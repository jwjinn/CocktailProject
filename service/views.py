from django.shortcuts import render

# Create your views here.

# 메인화면
def mainView(request):
    return render(request, 'service/index.html')

# 내가 마신 칵테일과 비슷한 칵테일은?
def cluster(request):
    return render(request, 'service/cluster.html')

# 지금 만들 수 있는 칵테일은?
def ingredient(request):
    return render(request, 'service/ingredient.html')

# 이 칵테일의 이름은?
def image(request):
    return render(request, 'service/image.html')

# 칵테일 사진 화풍변경
def changeImage(request):
    return render(request, 'service/changeImage.html')

# 사용 기술
def tech(request):
    return render(request, 'service/tech.html')

# 주변 칵테일 바의 위치는?
def barLocation(request):
    return render(request, 'service/barLocation.html')





