from django.shortcuts import render
from .forms import VideoForm
import cv2
import face_recognition
import numpy as np
import os

def process(file):
    path='images'
    images=[]
    classnames=[]
    mylist=os.listdir(path)
    print(mylist)

    for cl in mylist:
     curimg=cv2.imread(f'{path}/{cl}')
     images.append(curimg)
     classnames.append(os.path.splitext(cl)[0])
    
    print(classnames)  
    #print(images)  

    def encode(img):
     encli=[]
     for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        enc=face_recognition.face_encodings(img)[0]
        encli.append(enc)
     return encli
    encknown=encode(images)    
    print(len(encknown))

   

    cap = cv2.VideoCapture('videos/' + str(file))# capture video from file

    success,imgS=cap.read()

    # imgS = cv2.resize(img, (0, 0), None, 0.25, 0.5)
    # imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    # print(imgS)

    facescurframe=face_recognition.face_locations(imgS)
    #print(facescurframe)
    enccurframe=face_recognition.face_encodings(imgS,facescurframe)
    #print(len(encknown))

    for enco,face in zip(enccurframe,facescurframe):
        #print(len(encknown))
        if len(encknown) > 0:  # check if encknown is not empty
            matches=face_recognition.compare_faces(encknown,enco)
            facedis=face_recognition.face_distance(encknown,enco)
            print(facedis)
            matchindex=np.argmin(facedis)
            print(matchindex)

            if matches[matchindex]:
                name=classnames[matchindex].upper()
                print(name,'matched')
                cv2.putText(imgS,f'{name} matched',(20,170),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)

            else:
                print('encknown is empty')
                cv2.putText(imgS,f'Not matched',(20,170),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
                
    # display the frame
    imgS=cv2.resize(imgS,(640,480))
    cv2.imshow('Frame', imgS)
    cv2.waitKey(20000)
    
   

    

def upload(request):
    
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            name = form.cleaned_data['file']
            print(name)
            form.save()
            # Process the uploaded video
            process(name)
            return render(request, 'upload.html', {'form': form, 'message': 'Image processed successfully!'})
    else:
        form = VideoForm()
    return render(request, 'upload.html', {'form': form})
