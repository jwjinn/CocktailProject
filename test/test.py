#
# email = [['jwjinn@gmail.com'], ['example@gmail.com']]
#
# print(len(email))
# print(type(email))
# print(email.pop())
#
# if 4 >3:
#     print('true')

# # fs = FileSystemStorage(location='/home/joo/images', base_url='/home/joo/images')


import os

file_list = os.listdir('/home/joo/images')

print(file_list)
print(len(file_list))
