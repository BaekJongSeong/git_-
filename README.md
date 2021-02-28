# darknet-yolov4

This is a project where I participated in "the online hackathon for Dynamic Objects Detection" hosted by the Ministry of Science_ICT, NIA and hosted by Aimmo.

If the hackathon is divided into 3 part("the idea part, the technology part, and the commercialization part"), this part is a repository corresponding to the technology part and explains tensorflow-yolov4.

이 프로젝트는 제가 과학기술정보통신부 및 NIA한국정보화진흥원이 주관하고, Aimmo가 주최하는 동적객체인지 온라인 해커톤에 참가했던 프로젝트입니다.

해커톤을 3부분(아이디어 파트와 기술파트, 사업화파트)로 나누어보면, 이 부분은 기술파트에 해당하는 repository이며 yolov4에 대해서 설명하고 있습니다.

------
# 프로젝트 전체 개요

자율주행차량을 통해 제공되는 10329장의 image 데이터(공단에서 지정해준 사고다발지역)와 그에 해당하는 10329개의 json파일을 이용

object detection을 진행하기 위한 인공지능 모델 학습 후 test image 데이터들에 대한 모델 정확도를 테스트하는 형식

실습환경
1. Python 3.6.9 + GPU Tesla V100 + Ram 25.5GB 사양 사용을 위해 colab 사용

---
---
### + 1.dataset은 따로 다운받아야함 (github에 용량 제한)
	+ 다운 후 /darknet/bin/darknet/dynamic에 위치(png , txt 둘다 함꼐 하나의 폴더안에)
### + 2.경로는 지정 파일 위치로 사용자별로 바꿔야함
### + 3.weights파일도 따로 다운받아야 함(github에 용량 제한)
	+ 다운 후 darknet/ sia에 놓기
---
---
git clone하기 전에 다운받아야할 파일 + 명령어
```bash
#git clone
%cd /content/
!git clone https://github.com/AlexeyAB/darknet.git

# 3. 다운받을 yolov4.weights 파일
Download yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT
https://drive.google.com/file/d/1Y5JP2bn2I-Woqwsi-qhSs3WmGtKlwmov/view

#학습 전 권한 변경
#darknet 권한 변경 및 실행 테스트
%cd /content/drive/My\ Drive/Colab\ Notebooks/darknet/bin/darknet
!chmod +x ./darknet
!./darknet detector

#학습 명령-mAP통한
!./darknet detector train sia/sia2.DATA sia/sia.cfg sia/sia.137 -map -dont_show

#이어갈 학습 명령-backup파일에 들어 있음
!./darknet detector train sia/sia2.DATA sia/sia.cfg backup_01/sia_last.weights -map -dont_show

#detect명령
!./darknet detector test sia/sia2.DATA sia/sia.cfg backup_01/sia_best.weights /content/ab.png

```
------
전체 과정
### 1. 데이터 전처리
### 2. Config.py 파일 구성 및 hyperparameter수정
### 3. 모델 학습 train.py
### 4. detect.py를 통한 예측
---
+ 데이터 전처리
  + Aimmo측에서 제공해준 전체 데이터 중에서 정확한 데이터셋 확보를 위해 10329개의 json 파일 중, 365개의 empty annotation으로 이루어진 json 파일 제거. 
  + 최종 9964개의 data로 추림. (총 분류하려는 class 개수 3개로 조정: 자동차 사람 이륜차)
  + mAP 정확도 향상을 위해 epoch를 class 개수 *20000으로 총 60000번 지정. (최종 학습완료된 epoch는 12000번)
  + 데이터가 방대하기 때문에 Image argumentation을 적용하지 않음. (똑같은 사진을 회전과 좌우 반전을 부여하여 data 개수 늘림으로써 모델 성능이 정확)|
  + 각 bounding box마다의 좌표들을 image size 1920x1080에 대한 0~1 사이의 범위로 scaling 하는 과정 거침
  ### + Yolo(You Only Look Once)를 사용하기 위해서 json 파일을 txt 파일로 변환.

+ darknet yolov4 핵심 구성 파일
 + sia
	+ data
	+ names
	+ cfg
	+ weight
 + dynamic
	+ train_.txt
	+ png
	+ txt
---
 + cfg 파일
    + Subdivision = 64 (1 epoch 당 batch size)
    + Width & height = 416
    + momentum=0.949 (optimazation으로 과거 이동 방식 기억, 가중치 계속 부여)
    + channels=3	(1 epoch 당 3개의 channels)
    + learning_rate=0.001(학습 속도가 너무 느려도, 빨라도 안됨)
    + CNN activation=mish (relu activation보다 부드럽게 올라가는 gradient)
    + 마지막 부분의 CNN activation = linear
    + YOLO layer은 이미지의 feature들을 뽑고 난 후에 실질적인 prediction을 하는 layer. 
    + Mask: 총 9개의 anchor이 정의되어 있는데, 그 중 mask에 적혀있는 tag에 해당하는 anchor들만 사용.(ex) anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401)

+ Sia 파일
	+ .data: class 개수와 train/valid 데이터 셋 및 names 파일 위치 정보
	+ .names: class들의 이름: car, two-wheeled vehicle, pedestrian
	+ .weights: pre-trained 가중치 파일
	+ .cfg: batch, subdivision크기와 학습 이미지 크기 조정, hyperparameter 조정, learning 	        rate와 epoch, [convolutional neural network] 및 [yolo network]의 filter와 	   	        activation 조정, max pooling size 조정

+ Dynamic 파일
	+ train_.txt: 학습 image data의 파일 위치(dynamic/01.학습데이터자료/OBJ00013_PS3_K3_NIA0078.png)
	+ .png: 학습 이미지 파일
	+ .txt: 이미지에 해당하는 bounding box 좌표 파일


+ 생성된 모델을 통한 detect
