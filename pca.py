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
	W = normalize(top_evector)
	print(W.shape)
	return W

def rebuild(X,W,X_col):
	X0=np.mean(X,axis=1)
	d=W.shape[1]
	s=0
	for i in range(d):
		s+=sum(W[:,i]*(X[:,X_col-1]-X0))*W[:,i]
	return s+X0;

def reShow(imglist,d,x_col,N):
	X=readimg(imglist,N)
	W=PCA(X,d)
	Y=rebuild(X,W,x_col)
	Y=Y.reshape(50,40)
	image=Image.fromarray(Y)
	image.show()

if __name__=="__main__":
	imglist=[]
	for filename in os.listdir(r"AR"):
		imglist.append(filename)
#imglist存储图片的名称，150表示要降到的维数，2表示第2列，30是学习的图片数目
	reShow(imglist,150,2,30)

