from django.db import models as djmodels
from collections.abc import Iterable
import sys 
import os
#sys.path.append(os.path.abspath("\Users\Claudia\Desktop\project\gdp.py"))

import analyze.gdp as gdp



class PulseProfile(djmodels.Model):
    
    #creo la tabella e inserisco le colonne con i relativi nomi/key
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
    
    dedisp_profile=djmodels.BinaryField()
    allprofs=djmodels.BinaryField()
    disp_profile=djmodels.BinaryField()
    
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
    
    path_img=djmodels.FilePathField(path=None, match=None, recursive=False, max_length=100)
    bin_img=djmodels.BinaryField()
    
    
    def __str__(self):
        return self
    
    def restituisci_img(idn):
        return PulseProfile.objects.filter(id=idn).get().bin_img
    
    def restituisci_path(idn):
        x=PulseProfile.objects.filter(id=idn)
        y=x.get().path_img
        return y
    
    
    
    
    
    