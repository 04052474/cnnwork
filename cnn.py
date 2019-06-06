import pandas as pd

data=pd.read_csv('pokemon.csv')

print (data.head())


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
files= os.listdir('images')
label=[]
for i in range(len(files)):
    label.append(files[i].replace('.jpng'))#取得圖片名



img=[]
for i in range(len(files)):
    path=os.path.join('images',files[i])
    imgpic = Image.open(path) # 這時候還是 PIL object
    imgpic=imgpic.convert('RGB')
    img.append(np.array(imgpic))



#plt.imshow(img2)
#plt.show()
dataname=pd.DataFrame(data['Name'])    
dataname=pd.get_dummies(dataname)

for i in range(len(label)):
    label[i]==dataname

train_data=img[0:]
test_data=img[0:400]