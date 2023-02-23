import pandas as pd
import re
import os
from Transforms.FilteringTransforms import FilteringTransforms
nums = []
names = []
transform = FilteringTransforms()
ProjectDir = os.path.dirname(os.path.realpath(__file__))
Numbers = ["1","2","3","4","5","6","7","8","9","0"]
df = pd.read_csv(os.path.join(ProjectDir,'Datasets/TTSMetadata.csv'),sep = '\t')
for column in df["000000_RUSLAN|С тревожным чувством берусь я за перо."]:
    arr = column.split("|")
    nums.append(arr[0])
    arr[1] = arr[1].lower()
    arr[1] = re.sub(r'[^\w\s]','', arr[1])
    for num in Numbers:
        if num in arr[1]:
            arr[1] = arr[1].replace(num,"")
    for word in arr[1].split():
        if not word in names:
            names.append(word)
DatasetFiles = list(os.walk(os.path.join(ProjectDir,"Datasets/ASRDataset"),topdown=True))
for (root,dirs,files) in DatasetFiles:
    for file in files[:4096]:
        if file.endswith('.txt'):
            file = open(os.path.join(root,file),'r+',encoding="utf-8")
            DataFile = file.read()
            DataFile= DataFile.lower()
            DataFile = re.sub(r'[^\w\s]','', DataFile)
            for word in DataFile.split():
                if not word in names:
                    names.append(word)
            file.close()
    # names.append(arr[1].split())
# print(nums[:10])
# print(names)
with open(os.path.join(ProjectDir,'Datasets/RussianDict.txt'),"a+",encoding="utf-8") as file:
    for word in names:
        file.write(word + "\n")
