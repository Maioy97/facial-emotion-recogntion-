import numpy as np
import datetime 
import cv2
import os

def train():
	train_path = "/content/UTKFace"
	names = os.listdir(train_path)
	train_labels = []
	h_features = []  # hog feature collection
	hog = cv2.HOGDescriptor()
	svm = cv2.ml.SVM_create()
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setKernel(cv2.ml.SVM_RBF)  # cv2.ml.SVM_LINEA

	for name in names:
		nameparts = name.split('_')
		train_labels.append(int(nameparts[1]))
		# hog
		train_data = cv2.imread(train_path + '/' + name)
		train_data = cv2.resize(train_data, (64, 128), interpolation=cv2.INTER_AREA)
		h = hog.compute(train_data)
		h = h.ravel()
		h_features.append(h)

	print("loading and feature extraction done", datetime.datetime.now())
	train_features = np.float32(h_features)
	train_labelsnp = np.array(train_labels)
	svm.train(train_features, cv2.ml.ROW_SAMPLE, train_labelsnp)
	print("training done", datetime.datetime.now())
	svm.save("genderDetectionModel.xml")
	print("model saved", datetime.datetime.now())

def test():
	print("start:", datetime.datetime.now())
	testpath = "/content/crop_part1"
	names = os.listdir(testpath)
	test_labels = []
	h_features = []  # hog feature collection
	hog = cv2.HOGDescriptor()
	svm = cv2.ml.SVM_create()
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setKernel(cv2.ml.SVM_RBF)  # cv2.ml.SVM_LINEA

	for name in names:
		nameparts  = name.split('_')
		test_labels.append(int(nameparts [1]))
		# hog
		test_data = cv2.imread(testpath + '/' + name)
		test_data = cv2.resize(test_data, (64, 128), interpolation=cv2.INTER_AREA)
		h = hog.compute(test_data)
		h = h.ravel()
		h_features.append(h)
	print("loading and feature extraction done", datetime.datetime.now())
	test_features = np.float32(h_features)
	test_labelsnp = np.array(test_labels)
	svm_classifier = cv2.ml.SVM_load("genderDetectionModel.xml")
	correct = 0
	for i in range(test_labels.__len__()):
		response = svm_classifier.predict(np.array(test_features[i]).reshape(1, -1))
		if response[1].ravel() == test_labelsnp[i]:
		  correct += 1
	result = correct / test_labels.__len__()
	print(result)

