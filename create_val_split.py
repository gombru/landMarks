import random
import os

imgs = []
for file in os.listdir("../../hd/datasets/landmarks_recognition/img_train/"):
    imgs.append(file[:-4])


print "Img filenames read"
print len(imgs)

lines = []

with open("../../hd/datasets/landmarks_recognition/splits/train.csv", 'r') as f:
    for l in f:
        if l.split(',')[0].strip("\"") in imgs:
            lines.append(l)

print len(lines)
random.shuffle(lines)
print "lines[0]: " + str(lines[0])

val = open("../../hd/datasets/landmarks_recognition/splits/myVal.csv", 'w')
train = open("../../hd/datasets/landmarks_recognition/splits/myTrain.csv", 'w')

for i, l in enumerate(lines):
    # print l
    if i < int(0.05 * len(lines)):
        val.write(l)
    else:
        train.write(l)