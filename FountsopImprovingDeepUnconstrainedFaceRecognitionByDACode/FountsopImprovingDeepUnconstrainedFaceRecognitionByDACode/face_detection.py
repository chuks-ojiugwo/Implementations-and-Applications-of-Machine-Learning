def get_face_from_image(img, detector_path, dim=(100,100)):
	face_detector = dlib.cnn_face_detection_model_v1(detector_path)
	detected_faces, it= face_detector(img, 1), 0
	if len(detected_faces) != 0:
		d=detected_faces[0].rect
		imgf=img[d.top():d.bottom(),d.left():d.right()]
		if(imgf.shape[0]>1 and imgf.shape[1]>1):
			imgf= cv2.resize(imgf,dim)
			return imgf, len(detected_faces)
	return None, 0





if __name__ == '__main__':
	import cv2
	import os
	import time
	import dlib
	imageRoot ="LFW_OBJ"
	saveRoot = "LFW_AUGMENTED"
	if not os.path.exists(saveRoot):
	    os.makedirs(saveRoot)
	detector_path = "mmod_human_face_detector.dat"

	rootContent =[d for d in os.listdir(imageRoot)]
	for imageDir in rootContent:
		image_folder =imageRoot+"/"+imageDir
		save_folder=saveRoot+"/"+imageDir
		if not os.path.exists(save_folder):
			os.mkdir(save_folder)
		fileListe = os.listdir(image_folder)
		imgPaths=[x for x in fileListe if (x.endswith(".pgm") or x.endswith(".jpg"))]
		imgPaths =[d for d in imgPaths if d not in os.listdir(save_folder)]
		for imgP in imgPaths:
			img = cv2.imread(image_folder+"/"+imgP)
			img, nbf= get_face_from_image(img, detector_path, dim=(100,100))
			if(nbf!=0):
				cv2.imwrite(save_folder+"/"+imgP, img)


