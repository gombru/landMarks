import random

lines = []
with open("../../../hd/datasets/landmarks_recognition/splits/train.csv",'r') as f:
    for l in f:
        lines += l

random.shuffle(lines)

val =  open("../../../hd/datasets/landmarks_recognition/splits/myVal.csv",'w')
train =  open("../../../hd/datasets/landmarks_recognition/splits/myTrain.csv",'w')

for i,l in enumerate(lines):
    if i < int(0.05 * len(lines)):
        val.write(l)
    else:
        train.write(l)