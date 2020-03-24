

#from pkgutil import get_data as get_data
#from astropy.table import Table


%load_ext autoreload
%autoreload 2
import polls2.gendisp as gendisp
from django.db import models as djmodels
import polls2.models as models
from polls2.models import PulseProfile
import sys
from PIL import Image
from io import BytesIO
info=gendisp.generate_dispersed_profile()
info_dict=gendisp.print_info_to_db(info)
plt=gendisp.plot_profile(info)
p=PulseProfile(**info_dict)
p.save()
x=p.id
plt.savefig('C:/Users/Viviana/Desktop/mysite2/image/fig %d.png' % x)
PulseProfile.objects.filter(id=x).update(immagine="C:/Users/Viviana/Desktop/mysite2/image/fig %d.png" %x)
with open("C:/Users/Viviana/Desktop/mysite2/image/fig %d.png" %x,'rb') as file: blobData= file.read()
PulseProfile.objects.filter(id=x).update(bin_immagine=blobData)

#img.b64encode(bin_immagine=blobData).decode()
#bin_immagine(dr.GetSqlBytes(dr.GetOrdinal("C:/Users/Viviana/Desktop/mysite2/image/fig %d.png")).Buffer)

#bin_immagine.decode('base64')

z=PulseProfile.dammi_blob(x)
stream = BytesIO(z)
image = Image.open(stream).convert("RGBA")
stream.close()
image.show()


.

#byte[blob] = Convert.FromBase64String(z)
#File.WriteAllBytes(@"C:/Users/Viviana/Desktop/fic.jpg", blob)

#with open(r"image path.jpg","rb") as f:
   # z=f.read()
    #print(z)
#z = io.BytesIO(b64decoded_frame)


info_dict.pop("dedisp_profile")
info_dict.pop("allprofs")
info_dict.pop("disp_profile")