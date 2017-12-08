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

def get_template(img_num,dem,sample):
	imglist1=[]
	#imglist存储所有图片的名称
	for filename in os.listdir(r"AR"):
		imglist1.append(filename)
	temp_col=0     #选择需要降维的图片
	matrix_X=readimg(imglist1,img_num)
	W=PCA(matrix_X,dem)
	template=[]#test为作为对照的模板
	for k in range(len(sample)):
		temp_col=sample[k]
		get_Y=getY(matrix_X,W,temp_col)
		template.append(get_Y)
	return matrix_X,W,template

def group_distance(matrix_X,W,test_col,template):
	test=[]
	test=getY(matrix_X,W,test_col)
	distance=[]
	for i in range(len(template)):#len(template)
		distance.append(img_error(test,template[i]))
	return distance

if __name__ =="__main__":
	img_num=200#学习的图片数量
	dem=200   #降维的维度
#	test_col=16
	sample=[0,14,28,42]
	matrix_X,W,template=get_template(img_num,dem,sample)
	yes=0
	no=0
	for i in range(len(sample)):#group:0,1,2,3
		print("group:",i)
		for k in range(i*14+1,i*14+14):
			#testx=test(img_num,dem,sample,k)
			testx=group_distance(matrix_X,W,k,template)
			min_testx=np.argsort(testx)[0]
			print("min_testx:",min_testx)
			if min_testx == i:
				yes+=1
			else:
				no+=1
	print("参数：img_num||dem||template:",img_num,dem,sample)
	print("正确识别：",yes)
	print("错误识别：",no)
	print("识别准确率：",yes/(yes+no))
	



