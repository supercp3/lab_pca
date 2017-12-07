from PIL import Image
import os
import numpy as np
from sklearn.preprocessing import normalize

def readimg(imglist,num_pic):
	path="AR\\"+imglist[0]
	img=Image.open(path)
	x=np.array(img)
	m,n=x.shape
	dim=m*n
	p=x.reshape(dim,1)
	for i in range(1,num_pic):
		path="AR\\"+imglist[i]
		img=Image.open(path)
		x=np.array(img)
		x=x.reshape(dim,1)
		p=np.column_stack((p,x))
	return p

def PCA(X,d):
	x0=np.mean(X,axis=1)
	x1=X-x0.reshape((2000,1))
	print(x1.shape)
	m=x1.shape[1]
	E=(1/m)*np.dot(x1,x1.T)
	e_value,e_vector=np.linalg.eig(E)
	sorted_indices = np.argsort(e_value)
	top_evector = e_vector[:,sorted_indices[:-d-1:-1]]
	core=sum(sorted(e_value)[:-d-1:-1])/sum(e_value)#
	print(core)
	W = normalize(top_evector)
	return W
#重构图像
def reBuild(X,W,X_col):
	X0=np.mean(X,axis=1)
	d=W.shape[1]
	s=0
	#Y(dx1)=W[:,i]*(X[:,X_col-1]-X0
	for i in range(d):
		s+=sum(W[:,i]*(X[:,X_col]-X0))*W[:,i]
	return s+X0;

def reShow(Y):
	Z=Y.reshape(50,40)
	image=Image.fromarray(Z)
	image.show()

def reBuild_error(img_orign,img_rebuild):
	n=img_orign.shape[0]
	sx=0
	for i in range(n):
		sx+=np.square(img_orign[i]-img_rebuild[i])
	res=np.sqrt(sx)
	return res


if __name__=="__main__":
	imglist=[]
	#imglist存储所有图片的名称
	for filename in os.listdir(r"AR"):
		imglist.append(filename)
	#d降维后的维度，x_col降维的图片，N学习的图片数量
	d,x_col,N=1500,0,30
	img_matrix=readimg(imglist,N)
	W_img=PCA(img_matrix,d)
	#重建后的图像
	img_rebuild=reBuild(img_matrix,W_img,x_col)
	reShow(img_rebuild)
	#原始图像
	img_orign=img_matrix[:,x_col-1]
	#计算重建误差
	results=reBuild_error(img_orign,img_rebuild)
	print("重建误差为：\n",results)



