%load_ext autoreload
%autoreload 2

import analyze.gdp as gdp
from django.db import models as djmodels
import analyze.models as models
from analyze.models import PulseProfile
import sys
from PIL import Image
from io import BytesIO
#from astropy.table import Table


info=gdp.generate_dispersed_profile()
info_dict=gdp.print_info_to_db(info)
p=PulseProfile(**info_dict)
p.save()
plt=gdp.plot_profile(info)
idn=p.id
txt= "C:/Users/Claudia/Desktop/project/Fig/plot{}.png"
plt.savefig(txt.format(idn))
PulseProfile.objects.filter(id=idn).update(path_img="C:/Users/Claudia/Desktop/project/Fig/plot%d.png" %idn)
with open("C:/Users/Claudia/Desktop/project/Fig/plot%d.png" %idn, 'rb') as file: blobData = file.read()
PulseProfile.objects.filter(id=idn).update(bin_img=blobData)
x=PulseProfile.restituisci_path(idn)
y=PulseProfile.restituisci_img(idn)
conv = BytesIO(y)
imag = Image.open(conv).convert("RGBA")
conv.close()
imag.show()




# p=PulseProfile.objects.filter(id=idn)
# y=p.get().bin_img
# z=p.get().path_img

#q.path_img = "C:/Users/Claudia/Desktop/project/Fig/plot.png"
# info_dict.pop("dedisp_profile")
# info_dict.pop("allprofs")
# info_dict.pop("disp_profile")
# p=PulseProfile(**info_dict)
# p.save()