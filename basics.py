import cv2
import face_recognition

# Loading training image
image = face_recognition.load_image_file('images/test1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Loading test image
imageTest = face_recognition.load_image_file('images/test2.jpg')
imageTest = cv2.cvtColor(imageTest, cv2.COLOR_BGR2RGB)

# Getting face location of training image
faceLoc = face_recognition.face_locations(image)[0]
encodeImage = face_recognition.face_encodings(image)[0]
# print(faceLoc)      # returns a tuple of 4 values x1,x2,y1,y2
cv2.rectangle(image, (faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]), (255,0,255), 2)

# Test Image
faceLocTest = face_recognition.face_locations(imageTest)[0]                 # Just to see where the face is
encodeImageTest = face_recognition.face_encodings(imageTest)[0]
# print(faceLoc)      # returns a tuple of 4 values x1,x2,y1,y2
cv2.rectangle(imageTest, (faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]), (255,0,255), 2)

results = face_recognition.compare_faces([encodeImage], encodeImageTest)
# less is the face distance more the face is matching
faceDis = face_recognition.face_distance([encodeImage], encodeImageTest)
print(results)
print(faceDis)

cv2.putText(imageTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

cv2.imshow('Ayan Khan', image)
cv2.imshow('Ayan Khan Test', imageTest)
cv2.waitKey(0)
