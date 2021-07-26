from call_model import *
import os

class_name = ['Speed limit (20km/h)',
'Speed limit (30km/h)',
'No passing for vehicles over 3.5 metric tons',
'Right-of-way at the next intersection',
'Priority road',
'Yield',
'Stop',
'No vehicles',
'Vehicles over 3.5 metric tons prohibited',
'No entry',
'General caution',
'Dangerous curve to the left',
'Speed limit (50km/h)',
'Dangerous curve to the right',
'Double curve',
'Bumpy road',
'Slippery road',
'Road narrows on the right',
'Road work',
'Traffic signals',
'Pedestrians',
'Children crossing',
'Bicycles crossing',
'Speed limit (60km/h)',
'Beware of ice/snow',
'Wild animals crossing',
'End of all speed and passing limits',
'Turn right ahead',
'Turn left ahead',
'Ahead only',
'Go straight or right',
'Go straight or left',
'Keep right',
'Keep left',
'Speed limit (70km/h)',
'Roundabout mandatory',
'End of no passing',
'End of no passing by vehicles over 3.5 metric tons',
'Speed limit (80km/h)',
'End of speed limit (80km/h)',
'Speed limit (100km/h)',
'Speed limit (120km/h)',
'No passing'
]

# Enter path of folder to test
image_path = 'images\\original'
for root, dirs, files in os.walk(image_path):
    model = load_model_weights("models/weights_cnn.hdf5")
    WIDTH, HEIGHT = 32, 32
    for i in range(len(files)):
        path = image_path+"\\"+files[i]
        img = image.load_img(path, target_size = (WIDTH, HEIGHT))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        res = model.predict(img)
        Ypred = np.argmax(res, axis=1)
        print(class_name[Ypred[0]],"\t",files[i],"\t",Ypred[0])