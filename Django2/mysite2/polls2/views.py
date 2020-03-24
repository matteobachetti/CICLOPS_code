from django.shortcuts import render
from pkgutil import get_data as get_data
from django.http import HttpResponse


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

#def home(request):
#obj = BlobImg.objects.get(id=1)
    #image_data = base64.b64encode(obj.imm_photo)

    #data = {
        #'news': News.objects.all(),
        #'title': 'immagine',
       # imgs' : image_data
    #}
    #return render image_data
#def get_data(self, column):
   # query = UserData.objects.get(Username="Stack")
   # return getattr(query, column)
#def get_data(self, column):
    #query = UserData.objects.get(Username="")
   # return getattr(query, column)
#class_name.get_data('email')
PulseProfile.get_data('binary')