from pca import *
import os

def getY(matrix_X,w,col):
	x0=np.mean(matrix_X,axis=1)
	nx=w.shape[1]
	matrix_Y=[]
	sx=0
	for i in range(nx):
		sx=sum(w[:,i]*(matrix_X[:,col]-x0))
		matrix_Y.append(sx)
	return matrix_Y
def img_error(img_orign,img_rebuild):
	n=len(img_orign)
	sx=0
	for i in range(n):
		sx+=np.square(img_orign[i]-img_rebuild[i])
	res=np.sqrt(sx)
	return res

if __name__ =="__main__":
	imglist1=[]
	#imglist存储所有图片的名称
	for filename in os.listdir(r"AR"):
		imglist1.append(filename)
	img_num=50#学习的图片数量
	dem=200   #降维的维度
	temp_col=0     #选择需要降维的图片
	test_col=16
	matrix_X=readimg(imglist1,img_num)
	W=PCA(matrix_X,dem)
	sample=[0,14,28,42]
	template=[]#test为作为对照的模板
	for k in range(len(sample)):
		temp_col=sample[k]
		get_Y=getY(matrix_X,W,temp_col)
		template.append(get_Y)
	#print(template)
	test=getY(matrix_X,W,test_col)
	distance=[]
	for i in range(len(sample)):
		distance.append(img_error(test,template[i]))
	print(distance)

