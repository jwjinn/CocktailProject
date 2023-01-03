from django.shortcuts import render
from django.shortcuts import render, redirect
from django.http import JsonResponse
# Create your views here.


def index(request):

    if request.method == 'GET':
        return render(request, 'index.html')




def register(request):

    if request.method == 'GET':
        return render(request, 'register.html')

    if request.method == 'POST':
        print(request.POST['email'])
        return redirect('index')

