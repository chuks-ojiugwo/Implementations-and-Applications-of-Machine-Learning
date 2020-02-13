#########################################IMG###########DATA AUGMENTE####################################

def process_from_obj_list(objListe, image_folder, save_folder):
	"""
	This methode take a liste of obj fille and generates images in different light conditions.
	Input: 
		objListe: a liste of obj paths;
		image_folder: the obj files folder
		save_folder: the folder to save obj files
	"""
	for obj in objListe:
		startIM = time.time()
		img = np.zeros([h,h,3])
		liste=[np.zeros([h,h,3]) for i in sourceV]
		f = open(image_folder+"/"+obj, "r")
		texImg=obj.replace('.obj', '_texture.png')
		imgT = cv2.imread(image_folder+"/"+texImg)
		vertices=[]
		textures=[]
		faces=[]
		faceNorm=[]
		normals=[]
		
		#
		with open(image_folder+"/"+obj, "r") as filehandle:
			f = filehandle.readlines()
			for line in f:
				elts=line.split()
				if elts[0]=='v':
					vertices.extend([[float(elts[1]),float(elts[2]),float(elts[3])]])
				if elts[0]=='f':
					elt1=elts[1].split('/')
					v1=int(elt1[0])
					elt1=elts[2].split('/')
					v2=int(elt1[0])
					elt1=elts[3].split('/')
					v3=int(elt1[0])
					faces.extend([[v1,v2,v3]])
				if elts[0]=='vt':
					textures.extend([[float(elts[1]),float(elts[2])]])
		################FACE NORMAL######################################
		for f in faces:
			(v1,v2,v3)=(np.asarray(vertices[f[0]-1]),np.asarray(vertices[f[1]-1]),np.asarray(vertices[f[2]-1]))
			normal=np.cross(v2-v1,v3-v1)
			normals.extend([normal/np.linalg.norm(normal)])
		##############################END****FACE NORMAL#####################################
		faces=np.asarray(faces)
		faceNorm=np.asarray(normals)	
		vertices=np.asarray(vertices)
		textures=np.asarray(textures)
		############################VERTICE NORMAL###########################################
		verticesNor=np.array([[0.0,0.0,0.0] for x in vertices])
		fIndex=0
		for f in faces:
		    fn=faceNorm[fIndex]
		    for vf in f:
		        verticesNor[vf-1]=fn+verticesNor[vf-1]
		    fIndex=fIndex+1

		norm=np.linalg.norm(verticesNor, axis=1).reshape(-1,1)
		verticesNor= np.divide(verticesNor, norm)
		############################VERTICE NORMAL###########################################
		poids=np.array([[0.5,0.1,0.4],[0.1,0.9,0.0],[0.9,0.1,0.0],[0.5,0.2,0.3],[0.33,0.33,0.34],[0.3,0.5,0.2],[0.2,0.8,0.0],[0.5,0.25,0.25],[0.25,0.5,0.25],[0.25,0.25,0.5],[0.1,0.2,0.7],[0.3,0.5,0.2],[0.5,0.4,0.1]])


		for f in faces:
		    v1=vertices[f[0]-1]
		    v2=vertices[f[1]-1]
		    v3=vertices[f[2]-1]

		    p1,p2,p3=[h-int(v1[1]),int(v1[0])], [h-int(v2[1]),int(v2[0])], [h-int(v3[1]),int(v3[0])]
		    tt1=textures[f[0]-1]
		    tt2=textures[f[1]-1]
		    tt3=textures[f[2]-1]

		    lamb=np.ones(3)
		    t1,t2,t3=imgT[256-int(tt1[1]*256),int(tt1[0]*256)]*lamb[0], imgT[256-int(tt2[1]*256),int(tt2[0]*256)]*lamb[1], imgT[256-int(tt3[1]*256),int(tt3[0]*256)]*lamb[2]
		    for prq in poids:
		        c1=int(prq[0]*p1[0]+prq[1]*p2[0]+prq[2]*p3[0])
		        c2=int(prq[0]*p1[1]+prq[1]*p2[1]+prq[2]*p3[1])
		        img[c1,c2]=(prq[0]*t1 +prq[1]*t2 +prq[2]*t3)



		    lt=np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
		    for ii in range(len(sourceV)):
			    for i in range(0,3):
			        lt[i]=vertices[f[i]-1] - np.array(sourceV[ii])
			        lt[i]=lt[i]/np.linalg.norm(lt[i])
			        lamb[i]=np.vdot(lt[i],verticesNor[f[i]-1])
			    lamb=lamb*e
			    t1,t2,t3=imgT[256-int(tt1[1]*256),int(tt1[0]*256)]*lamb[0], imgT[256-int(tt2[1]*256),int(tt2[0]*256)]*lamb[1], imgT[256-int(tt3[1]*256),int(tt3[0]*256)]*lamb[2]
			    for prq in poids:
			        c1=int(prq[0]*p1[0]+prq[1]*p2[0]+prq[2]*p3[0])
			        c2=int(prq[0]*p1[1]+prq[1]*p2[1]+prq[2]*p3[1])
			        liste[ii][c1,c2]=0.5*img[c1,c2]+(prq[0]*t1 +prq[1]*t2 +prq[2]*t3)



		FImg=obj.replace('.obj', '')
		cv2.imwrite(save_folder+"/"+FImg+".png", img)
		for v in range(len(sourceV)):
			cv2.imwrite(save_folder+"/"+FImg+str(v)+".png", liste[v])

		end = time.time()
		print(end - startIM)

if __name__ == '__main__':
	import numpy as np
	import cv2
	import os
	import random , os
	import time
	import argparse
	parser = argparse.ArgumentParser(description='Data augmentation based on OBJ')
	parser.add_argument('-i', '--inputDir', default='lfwOBJ', type=str, help='path to the input directory, the OBJ files are stored.')
	parser.add_argument('-o', '--outputDir', default='AUGMENTED', type=str, help='path to the output directory')

	args=parser.parse_args()
	imageRoot = args.inputDir
	saveRoot =  args.outputDir
	rootContent=[d for d in os.listdir(imageRoot)]
	h=256
	e=1.2
	#set of liigth source point in the 3D space
	sourceV=np.array([[-100, 100, 20],[-60, 100, 20],[-20, 100,20],[0,100,20], [20,100,20],[40,100,20], [60,100,20],
	[80,100,20],[100, 100, 20],[125,100,20],[150,100,20], [175,100,20], [200,100,20], [250,100,20],
	[275,100,20],[300,100,20],[325,100,20],[350, 100, 20]])


	if not os.path.exists(saveRoot):
	    os.makedirs(saveRoot)
	for imageDir in rootContent:
		image_folder =imageRoot+"/"+imageDir
		save_folder=saveRoot+"/"+imageDir
		if not os.path.exists(save_folder):
			os.mkdir(save_folder)
		fl = os.listdir(image_folder)
		objNAME=[x for x in fl if x.endswith(".obj")]
		start = time.time()
		parImage(objNAME, image_folder, save_folder)
		end = time.time()