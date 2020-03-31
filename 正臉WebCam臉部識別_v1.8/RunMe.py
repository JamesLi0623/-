import face_recognition
import cv2
from james_module import addPerson

# 1. 這版是加速版，未來都以加速版為更新主體，慢速版若無特殊需要暫時不更新
# 2. 本版把索引資料的資料結構換成class，方便日後管理
# 3. API 中 addPerson() 參數資料有變更，且本版配合使用模型修正為新加入的參考資料會在陣列的最前端以提升命中率，務必整個 james_module.py 一起更新

# 宣告索引資料的資料結構類別，並建立索引資料串列 ref_list
class ref:
    name = str()
    face_encoding = list()
    def __init__(self, name, encoding):
        self.name = name
        self.face_encoding = encoding

ref_list = list()


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
james_image = face_recognition.load_image_file("james.jpg")
james_face_encoding = face_recognition.face_encodings(james_image)[0]
ref_list.insert(0, ref("James Li", james_face_encoding))

# Load a second sample picture and learn how to recognize it.
banana_image = face_recognition.load_image_file("banana.jpg")
banana_face_encoding = face_recognition.face_encodings(banana_image)[0]
ref_list.insert(0, ref("Banana Ding", banana_face_encoding))

# Load a third sample picture and learn how to recognize it.
single_image = face_recognition.load_image_file("single.jpg")
single_face_encoding = face_recognition.face_encodings(single_image)[0]
ref_list.insert(0, ref("Single Liu", single_face_encoding))

# Load a forth sample picture and learn how to recognize it.
sister_image = face_recognition.load_image_file("sister.jpg")
sister_face_encoding = face_recognition.face_encodings(sister_image)[0]
ref_list.insert(0, ref("Sister Wang", sister_face_encoding))


# 連續加入索引資料組，呼叫自訂函式 addPerson，將新增資料組加入
while True:
    want_add = input("Do you want add a person? [y/n] : ")
    if(want_add == "n"):
        break
    elif(want_add == "y"):
        file_name = input("File name : ")
        person_name = input("Person name : ")
        addPerson(ref_list, file_name, person_name)


# 將 ref_list 中的所有 face encodings 及 names 依序抽出建立串列
known_face_encodings = list()
for element in ref_list:
    known_face_encodings.append(element.face_encoding)
known_face_names = list()
for element in ref_list:
    known_face_names.append(element.name)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/2 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,0.45)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/2 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 195), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left - 1, bottom - 25), (right + 1, bottom + 10), (255, 0, 195), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom + 4), font, 0.88, (255, 255, 255), 1)


        # # Draw a label with a name up the face，畫在上面的版本
        # cv2.rectangle(frame, (left - 1, top - 25), (right + 1, top + 10), (255, 0, 195), cv2.FILLED)
        # font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(frame, name, (left + 6, top + 4), font, 0.9, (255, 255, 255), 1)


    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()