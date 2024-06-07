from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy

#load pre-trained yolov5n model
model = YOLO("./model/yolov5nu.pt")
names = model.names

# 이미지
ALLOWED_IMG_EXTENSIONS = {'png','jpg','jpeg, pdf'}
# 텍스트
# ALLOWED_TXT_EXTENSTIONS = {'txt','pdf', 'docx'}
# 비디오
ALLOWED_VD_EXTENSIONS = {'mp4','avi'}
# 오디오
# ALLOWED_AD_EXTENSIONS = {'mp3','m4a','wav'}


app = Flask(__name__)
CORS(app, resources={r"/labeling": {"origins": "*"}})

# filename : file.jpg, file.mp4
# 파일이 이미지, 텍스트, 비디오, 음성인지 확장자를 통해 구분하여 종류를 반환함.
def classify(filename):
    name, extend = filename.split('.')
    if extend in ALLOWED_IMG_EXTENSIONS:
        return "img"
    # elif extend in ALLOWED_TXT_EXTENSTIONS:
        return "txt"
    elif extend in ALLOWED_VD_EXTENSIONS:
        return "vd"
    # elif extend in ALLOWED_AD_EXTENSIONS:
        # return "ad"
    else:
        return "null"

def handleImg(file):
    # 이미지 클래스 이름 추출 ex) car, bus, tie
    upload_path = os.path.join('uploads', file.filename)
    result = model.predict(upload_path)
    kwd_i = []
    for frame_results in result:
        for box in frame_results.boxes:
            confidence = box.conf.numpy().tolist()[0]
            class_id = int(box.cls.numpy().tolist()[0])
            class_name = names[class_id]
            if 0 < confidence <= 1 and confidence > 0.5:
                kwd_i.append(class_name)
    kwd_i = list(set(kwd_i))
    return kwd_i
    
# def handleTxt(file):
    
def handleVd(file):
    kwd_v = []
    upload_path = os.path.join('uploads', file.filename)
    result = model.predict(upload_path)
    for frame_results in result:
        for box in frame_results.boxes:
            confidence = box.conf.numpy().tolist()[0]
            class_id = int(box.cls.numpy().tolist()[0])
            class_name = names[class_id]
            if 0 < confidence <= 1 and confidence > 0.5:
                kwd_v.append(class_name)
    kwd_v = list(set(kwd_v))
    return kwd_v

# def handleAd(file:)

# 타입에 따른 모델을 실행함
def getTag(file):
    extend = classify(file.filename)
    aiTag = None
    if extend == "img":
        aiTag = handleImg(file)
    # https://huggingface.co/jhgan/ko-sroberta-multitask
    # elif extend == "txt":
    #     aiTag = handleTxt(file)
    elif extend == "vd":
        aiTag = handleVd(file)
    # elif extend == "ad":
    #     aiTag = handleAd(file)
    else:
        aiTag = "Error"
    return aiTag

#파일 업로드 처리
@app.route('/labeling', methods=['POST'])
def labeling():
    file = request.files['file']
    # 예측에 사용될 파일을 저장할 경로
    upload_path = os.path.join('uploads', file.filename)
    file.save(upload_path)
    # 모델 예측
    ai_labels = getTag(file)
    print(ai_labels)
    res = {
        "ai_labels" : ai_labels
    }
    # 모델 예측에 사용된 데이터 삭제
    os.remove(upload_path)
    # 예측 값 전달
    return res



if __name__ == '__main__':
    app.run(port=8000, debug=True)
