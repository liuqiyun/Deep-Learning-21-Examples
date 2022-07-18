# 资源： https://blog.csdn.net/m0_55479420/article/details/115268470
import cv2
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle

from numpy import asarray
from PIL import Image

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine


def highlight_faces(image_path, faces):
    image = plt.imread(image_path)
    plt.imshow(image)
    ax = plt.gca()

    for face in faces:
        x, y, width, height = face['box']
        face_border = Rectangle((x, y), width, height,
                                fill=False, color='red')
        ax.add_patch(face_border)
    plt.show()


# image_path = '/home/liuqiyun/img/messi-1.jpeg'
# face_messi_1_image_path = '/home/liuqiyun/img/messi-face-1.jpeg'
# face_messi_2_image_path = '/home/liuqiyun/img/messi-face-2.jpeg'
# face_messi_4_image_path = '/home/liuqiyun/img/messi-face-4.jpeg'
# face_cr7_1_image_path = '/home/liuqiyun/img/cr7-face-1.jpeg'
# face_cr7_2_image_path = '/home/liuqiyun/img/cr7-face-2.jpeg'
# face_cr7_3_image_path = '/home/liuqiyun/img/cr7-face-3.jpeg'
image_path = 'img/messi-1.jpeg'
face_messi_1_image_path = 'img/messi-face-1.jpeg'
face_messi_2_image_path = 'img/messi-face-2.jpeg'
face_messi_4_image_path = 'img/messi-face-4.jpeg'
face_cr7_1_image_path = 'img/cr7-face-1.jpeg'
face_cr7_2_image_path = 'img/cr7-face-2.jpeg'
face_cr7_3_image_path = 'img/cr7-face-3.jpeg'


# 人脸检测：返回人脸坐标
image = plt.imread(image_path)
detector = MTCNN()
faces = detector.detect_faces(image)
for face in faces:
    print(face)

# 人脸检测：在图像中高亮显示
highlight_faces(image_path, faces)

# 人脸提取
def extract_face_from_image_method1(image_path, required_size=(224, 224)):
    image = plt.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)
    face_images = []
    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height

        face_boundary = image[y1:y2, x1:x2]

        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)
    return face_images


# https://www.cnpython.com/qa/1331421
# 需要用这个方法，否则调用模型特征值对比的时候可能会报错
def extract_face_from_image(image_path, required_size=(224, 224)):
    # load image and detect faces
    image = plt.imread(image_path)

    detector = MTCNN()
    faces = detector.detect_faces(image)

    # extract the bounding box from the requested face
    x1, y1, width, height = faces[0]['box']
    x2, y2 = x1 + width, y1 + height

    # extract the face
    face_boundary = image[y1:y2, x1:x2]

    image = cv2.resize(face_boundary, required_size)

    return image


extracted_face1 = extract_face_from_image_method1(face_messi_1_image_path)
plt.imshow(extracted_face1[0])
plt.show()
extracted_face4 = extract_face_from_image_method1(face_messi_4_image_path)
plt.imshow(extracted_face4[0])
plt.show()
extracted_face5 = extract_face_from_image_method1(face_cr7_1_image_path)
plt.imshow(extracted_face5[0])
plt.show()
extracted_face3 = extract_face_from_image(face_messi_1_image_path)
plt.imshow(extracted_face3[1])
plt.show()


def get_model_scores(faces):
    samples = asarray(faces, 'float32')

    samples = preprocess_input(samples, version=2)

    model = VGGFace(model='resnet50',
                    include_top=False,
                    input_shape=(224, 224, 3),
                    pooling='avg')

    return model.predict(samples)


def compare(similarity):
    if similarity <= 0.4:
        print("Faces Matched")
    else:
        print("Faces DO NOT Matched")


# 分别比较两张人脸照片，看是否相似
faces = [extract_face_from_image(image_path)
         for image_path in [face_messi_1_image_path, face_messi_1_image_path]]
model_scores = get_model_scores(faces)
similarity = cosine(model_scores[0], model_scores[1])
print("similarity(messi-1 vs messi-1) = " + str(similarity))
compare(similarity)


faces = [extract_face_from_image(image_path)
         for image_path in [face_messi_1_image_path, face_messi_4_image_path]]
model_scores = get_model_scores(faces)
similarity = cosine(model_scores[0], model_scores[1])
print("similarity(messi-1 vs messi-4) = " + str(similarity))
compare(similarity)

faces = [extract_face_from_image(image_path)
         for image_path in [face_messi_1_image_path, face_cr7_1_image_path]]
model_scores = get_model_scores(faces)
similarity = cosine(model_scores[0], model_scores[1])
print("similarity(messi-1 vs cr7-1) = " + str(similarity))
compare(similarity)

faces = [extract_face_from_image(image_path)
         for image_path in [face_cr7_1_image_path, face_cr7_2_image_path]]
model_scores = get_model_scores(faces)
similarity = cosine(model_scores[0], model_scores[1])
print("similarity(cr7-1 vs cr7-2) = " + str(similarity))
compare(similarity)

# if cosine(model_scores[0], model_scores[1]) <= 0.4:
# if similarity <= 0.4:
#     print("Faces Matched")
# else:
#     print("Faces DO NOT Matched")
