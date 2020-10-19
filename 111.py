import os

images = os.listdir('/home/zjx/roadside/coco128/images/val2017')
images = [os.path.join('/home/zjx/roadside/coco128/images/val2017', image) for image in images]
for image in images:
    idx = int(image.split('img')[1].split('.jpg')[0])
    source = image
    end = image.split('img')[0] + '%05d.jpg' % idx
    os.system('mv %s %s' % (source, end))
print('finished')