from django.db import models as djmodels
from collections.abc import Iterable
import polls2.gendisp as gendisp
import base64
import requests
      

class PulseProfile(djmodels.Model):
    pulse_freq=djmodels.FloatField(default=0)
    start_freq=djmodels.IntegerField(default=0)
    bandwidth=djmodels.IntegerField(default=0)
    nchan=djmodels.IntegerField(default=0)
    dm=djmodels.FloatField(default=0)
    amp=djmodels.FloatField(default=0)
    width=djmodels.FloatField(default=0)
    ph0=djmodels.FloatField(default=0)
    nbin=djmodels.IntegerField(default=0)
    noise_level=djmodels.FloatField(default=0)
    dedisp_profile=djmodels.BinaryField()# dati da convertire in binario perchè sono in ndarray
    allprofs=djmodels.BinaryField()# dati da convertire in binario perchè sono in ndarray
    disp_profile=djmodels.BinaryField()# dati da convertire in binario perchè sono in ndarray
    disp_z2=djmodels.FloatField(default=0)
    disp_z6=djmodels.FloatField(default=0)
    disp_z12=djmodels.FloatField(default=0)
    disp_z20=djmodels.FloatField(default=0)
    disp_H=djmodels.FloatField(default=0)
    dedisp_z2=djmodels.FloatField(default=0)
    dedisp_z6=djmodels.FloatField(default=0)
    dedisp_z12=djmodels.FloatField(default=0)
    dedisp_z20=djmodels.FloatField(default=0)
    dedisp_H=djmodels.FloatField(default=0)
    immagine=djmodels.FilePathField(path=None, match=None, recursive=False, max_length=100) #path
    ##allprofs1=djmodels.FilePathField(path=None, match=None, recursive=False, max_length=100)
    #disp_profile1=djmodels.FilePathField(path=None, match=None, recursive=False, max_length=100)
    bin_immagine=djmodels.BinaryField() #immagini in binario
    #bin_allprofs1=djmodels.BinaryField()
    #bin_disp_profile1=djmodels.BinaryField()
#class BlobImg(djmodels.Model):
    #imm_photo = djmodels.ImageField(upload_to="imm_photos", storage=DatabaseStorage() )
   
   # def get_data(self, x, bin_immagine):
        #query = PulseProfile.objects.filter(id=x)
        #return getattr(query, bin_immagine)
    def dammi_blob(x):
        blob = PulseProfile.objects.filter(id=x).get().bin_immagine
        return blob
   # def pass_image(z):
        #output = blob.b64encode(requests.get(z).content)
        #bin = "".join(format(ord(z), "b") for z in base64.decodestring(output))
        #return bin # or you could print it
    