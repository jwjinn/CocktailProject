import argon2.exceptions
from django.shortcuts import render
from django.shortcuts import render, redirect
from django.http import JsonResponse
from argon2 import PasswordHasher
from login.models import *

# Create your views here.
# https://arotein.tistory.com/22

def index(request):

    if request.method == 'GET':
        return render(request, 'index.html')

    if request.method == 'POST':

        email = request.POST['email']
        password = request.POST['password']

        print(email)
        print(password)


        login = Userregister.objects.filter(email=email)

        if( not login.exists()):
            print("아이다가 없다.")
            return redirect('index')


        dbPassword = login.values()[0]['password']

        try:
            if (PasswordHasher().verify(dbPassword, password)):
                print("비밀번호 일치한다.")



                return redirect('service/')

        except argon2.exceptions.VerifyMismatchError as e:
            print("비밀번호 일치 안함.")
            return redirect('index')


        return redirect('index')



def register(request):

    if request.method == 'GET':
        return render(request, 'register.html')

    if request.method == 'POST':
        email = request.POST['email']
        password = PasswordHasher().hash(request.POST['password'])
        name = request.POST['name']
        gender = request.POST['gender']
        age = int(request.POST['age'])
        phone_number = request.POST['phone_number']

        register = Userregister(email= email, password= password, name=name,
                                gender=gender, age=age, phone_number=phone_number)

        register.save()


        return redirect('index')

