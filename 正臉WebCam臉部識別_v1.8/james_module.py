import face_recognition
import cv2

# 宣告索引資料的資料結構類別
class ref:
    name = str()
    face_encoding = list()
    def __init__(self, name, encoding):
        self.name = name
        self.face_encoding = encoding


# 新增一組照片與姓名
def addPerson(ref_list, new_image_flie, new_name):
    # ref_list為參考資料串列，在 kernel 中同名；new為新加入的資料值。new_image_flie 記得加副檔名
    new_image = face_recognition.load_image_file(new_image_flie)
    new_face_encoding = face_recognition.face_encodings(new_image)[0]

    element = ref(new_name, new_face_encoding)
    ref_list.insert(0, element)

    return ref_list