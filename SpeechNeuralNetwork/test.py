import pandas as pd
nums = []
names = []
df = pd.read_csv('Datasets/TTSMetadata.csv',sep = '\t')
for column in df["000000_RUSLAN|С тревожным чувством берусь я за перо."]:
    arr = column.split("|")
    nums.append(arr[0])
    names.append(arr[1])
print(nums[:10])
# print(names)