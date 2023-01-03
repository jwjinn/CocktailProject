from django.shortcuts import render
from django.shortcuts import render, redirect
from django.http import JsonResponse
# Create your views here.


def index(request):

    if request.method == 'GET':
        return render(request, 'index.html')

    if request.method == 'POST':
        print(request.POST['email'])
        print(request.POST['password'])
        return redirect('index')




def register(request):

    if request.method == 'GET':
        return render(request, 'register.html')

    if request.method == 'POST':
        print(request.POST['email'])
        print(request.POST['password'])
        print(request.POST['name'])
        print(request.POST['gender'])
        print(request.POST['age'])
        print(request.POST['phone_number'])
        return redirect('index')

