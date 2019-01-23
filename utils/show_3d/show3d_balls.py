import numpy as np
import ctypes as ct
import cv2
import sys
import os
import pdb 
showsz=800
mousex,mousey=0.5,0.5
zoom=1.0
changed=True

# partseg colors
l1 = (0,187,255)
l2 = (60,255,0)
l3 = (255,68,0)
l4 = (187,0,255)
label_colors = [l1,l2,l3,l4]


def onmouse(*args):
	global mousex,mousey,changed
	y=args[1]
	x=args[2]
	mousex=x/float(showsz)
	mousey=y/float(showsz)
	changed=True
cv2.namedWindow('show3d')
cv2.moveWindow('show3d',0,0)
cv2.setMouseCallback('show3d',onmouse)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dll=np.ctypeslib.load_library(os.path.join(BASE_DIR, 'render_balls_so'),'.')

green=np.linspace(0,1,10000)
red=np.linspace(1,0,10000)**0.5
blue=np.linspace(1,0,10000)

def get2D(xyz, partseg=False, labels=None, ballradius=7, background=(0,0,0), showsz=400):

	# xyz=xyz-xyz.mean(axis=0)
	radius=((xyz**2).sum(axis=-1)**0.5).max()
	xyz/=(radius*2.2)/showsz

	if partseg:
		c0 = np.zeros((len(xyz),),dtype='float32')
		c1 = np.zeros((len(xyz),),dtype='float32')
		c2 = np.zeros((len(xyz),),dtype='float32')
		for i,l in enumerate(labels):
			c0[i] = label_colors[l-1][0]
			c1[i] = label_colors[l-1][1]
			c2[i] = label_colors[l-1][2]
		# if normalizecolor:
		# 	c0/=(c0.max()+1e-14)/255.0
		# 	c1/=(c1.max()+1e-14)/255.0
		# 	c2/=(c2.max()+1e-14)/255.0

		c0=np.require(c0,'float32','C')
		c1=np.require(c1,'float32','C')
		c2=np.require(c2,'float32','C')

	else:
		c0=np.zeros((len(xyz),),dtype='float32')+255
		c1=c0
		c2=c0

		c0=np.require(c0,'float32','C')
		c1=np.require(c1,'float32','C')
		c2=np.require(c2,'float32','C')


	show=np.zeros((showsz,showsz,3),dtype='uint8')
	def render():
		nxyz = xyz + [showsz/2,showsz/2,0]
		ixyz = nxyz.astype('int32')
		show[:]=background
		dll.render_ball(
			ct.c_int(show.shape[0]),
			ct.c_int(show.shape[1]),
			show.ctypes.data_as(ct.c_void_p),
			ct.c_int(ixyz.shape[0]),
			ixyz.ctypes.data_as(ct.c_void_p),
			c0.ctypes.data_as(ct.c_void_p),
			c1.ctypes.data_as(ct.c_void_p),
			c2.ctypes.data_as(ct.c_void_p),
			ct.c_int(ballradius)
		)

	render()
	return show

def get2Dtwopoints(xyz,xyz_pred,c0_pred=green,c1_pred=red,c2_pred=blue,background=(0,0,0),ballradius=7, showsz=400):

	# xyz=xyz-xyz.mean(axis=0)
	radius=((xyz**2).sum(axis=-1)**0.5).max()
	xyz/=(radius*2.2)/showsz

	# xyz_pred=xyz_pred-xyz_pred.mean(axis=0)
	radius_pred=((xyz_pred**2).sum(axis=-1)**0.5).max()
	xyz_pred/=(radius_pred*2.2)/showsz

	c0=np.zeros((len(xyz),),dtype='float32')+255
	c1=c0
	c2=c0

	c0=np.require(c0,'float32','C')
	c1=np.require(c1,'float32','C')
	c2=np.require(c2,'float32','C')

	c1_pred = c0
	c2_pred = c0
	c3_pred = c0
	c0_pred=np.require(c0_pred,'float32','C')
	c1_pred=np.require(c1_pred,'float32','C')
	c2_pred=np.require(c2_pred,'float32','C')

	show=np.zeros((showsz,showsz,3),dtype='uint8')

	def render():
		nxyz = xyz + [showsz/2,showsz/2,0]
		nxyz_pred = xyz_pred + [showsz/2,showsz/2,0]

		ixyz=nxyz.astype('int32')
		ixyz_pred=nxyz_pred.astype('int32')

		show[:]=background
		dll.render_ball(
			ct.c_int(show.shape[0]),
			ct.c_int(show.shape[1]),
			show.ctypes.data_as(ct.c_void_p),
			ct.c_int(ixyz.shape[0]),
			ixyz.ctypes.data_as(ct.c_void_p),
			c0.ctypes.data_as(ct.c_void_p),
			c1.ctypes.data_as(ct.c_void_p),
			c2.ctypes.data_as(ct.c_void_p),
			ct.c_int(ballradius)
		)
		dll.render_ball(
			ct.c_int(show.shape[0]),
			ct.c_int(show.shape[1]),
			show.ctypes.data_as(ct.c_void_p),
			ct.c_int(ixyz_pred.shape[0]),
			ixyz_pred.ctypes.data_as(ct.c_void_p),
			c0_pred.ctypes.data_as(ct.c_void_p),
			c1_pred.ctypes.data_as(ct.c_void_p),
			c2_pred.ctypes.data_as(ct.c_void_p),
			ct.c_int(ballradius)
		)

	render()
	return show

	

def showpoints_partseg(xyz,labels,c0=None,c1=None,c2=None,waittime=0,showrot=False,magnifyBlue=0,freezerot=False,background=(0,0,0),normalizecolor=True,ballradius=10):
	global showsz,mousex,mousey,zoom,changed
	# xyz=xyz-xyz.mean(axis=0)
	radius=((xyz**2).sum(axis=-1)**0.5).max()
	xyz/=(radius*2.2)/showsz
	c0 = np.zeros((len(xyz),),dtype='float32')
	c1 = np.zeros((len(xyz),),dtype='float32')
	c2 = np.zeros((len(xyz),),dtype='float32')
	for i,l in enumerate(labels):
		c0[i] = label_colors[l-1][0]
		c1[i] = label_colors[l-1][1]
		c2[i] = label_colors[l-1][2]
	if normalizecolor:
		c0/=(c0.max()+1e-14)/255.0
		c1/=(c1.max()+1e-14)/255.0
		c2/=(c2.max()+1e-14)/255.0
	c0=np.require(c0,'float32','C')
	c1=np.require(c1,'float32','C')
	c2=np.require(c2,'float32','C')

	show=np.zeros((showsz,showsz,3),dtype='uint8')
	def render():
		rotmat=np.eye(3)
		if not freezerot:
			xangle=(mousey-0.5)*np.pi*1.2
		else:
			xangle=0
		rotmat=rotmat.dot(np.array([
			[1.0,0.0,0.0],
			[0.0,np.cos(xangle),-np.sin(xangle)],
			[0.0,np.sin(xangle),np.cos(xangle)],
			]))
		if not freezerot:
			yangle=(mousex-0.5)*np.pi*1.2
		else:
			yangle=0
		rotmat=rotmat.dot(np.array([
			[np.cos(yangle),0.0,-np.sin(yangle)],
			[0.0,1.0,0.0],
			[np.sin(yangle),0.0,np.cos(yangle)],
			]))
		rotmat*=zoom
		nxyz=xyz.dot(rotmat)+[showsz/2,showsz/2,0]

		ixyz=nxyz.astype('int32')
		show[:]=background
		dll.render_ball(
			ct.c_int(show.shape[0]),
			ct.c_int(show.shape[1]),
			show.ctypes.data_as(ct.c_void_p),
			ct.c_int(ixyz.shape[0]),
			ixyz.ctypes.data_as(ct.c_void_p),
			c0.ctypes.data_as(ct.c_void_p),
			c1.ctypes.data_as(ct.c_void_p),
			c2.ctypes.data_as(ct.c_void_p),
			ct.c_int(ballradius)
		)

		if magnifyBlue>0:
			show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=0))
			if magnifyBlue>=2:
				show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=0))
			show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=1))
			if magnifyBlue>=2:
				show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=1))
		if showrot:
			cv2.putText(show,'xangle %d'%(int(xangle/np.pi*180)),(30,showsz-30),0,0.5,cv2.cv.CV_RGB(255,0,0))
			cv2.putText(show,'yangle %d'%(int(yangle/np.pi*180)),(30,showsz-50),0,0.5,cv2.cv.CV_RGB(255,0,0))
			cv2.putText(show,'zoom %d%%'%(int(zoom*100)),(30,showsz-70),0,0.5,cv2.cv.CV_RGB(255,0,0))
	changed=True
	while True:
		if changed:
			render()
			changed=False
		cv2.imshow('show3d',show)
		if waittime==0:
			cmd=cv2.waitKey(10)%256
		else:
			cmd=cv2.waitKey(waittime)%256
		if cmd==ord('q'):
			break
		elif cmd==ord('Q'):
			sys.exit(0)
		if cmd==ord('n'):
			zoom*=1.1
			changed=True
		elif cmd==ord('m'):
			zoom/=1.1
			changed=True
		elif cmd==ord('r'):
			zoom=1.0
			changed=True
		elif cmd==ord('s'):
			cv2.imwrite('show3d.png',show)
		if waittime!=0:
			break
	return cmd	


def showtwopoints(xyz,xyz_pred,c0_pred=green,c1_pred=red,c2_pred=blue,waittime=0,showrot=False,magnifyBlue=0,freezerot=False,background=(0,0,0),ballradius=10):

	global showsz,mousex,mousey,zoom,changed

	# xyz=xyz-xyz.mean(axis=0)
	radius=((xyz**2).sum(axis=-1)**0.5).max()
	xyz/=(radius*2.2)/showsz

	# xyz_pred=xyz_pred-xyz_pred.mean(axis=0)
	radius_pred=((xyz_pred**2).sum(axis=-1)**0.5).max()
	xyz_pred/=(radius_pred*2.2)/showsz

	c0=np.zeros((len(xyz),),dtype='float32')+255
	c1=c0
	c2=c0

	c0=np.require(c0,'float32','C')
	c1=np.require(c1,'float32','C')
	c2=np.require(c2,'float32','C')

	c1_pred = c0
	c2_pred = c0
	c3_pred = c0
	c0_pred=np.require(c0_pred,'float32','C')
	c1_pred=np.require(c1_pred,'float32','C')
	c2_pred=np.require(c2_pred,'float32','C')

	show=np.zeros((showsz,showsz,3),dtype='uint8')

	def render():
		rotmat=np.eye(3)
		if not freezerot:
			xangle=(mousey-0.5)*np.pi*1.2
		else:
			xangle=0
		rotmat=rotmat.dot(np.array([
			[1.0,0.0,0.0],
			[0.0,np.cos(xangle),-np.sin(xangle)],
			[0.0,np.sin(xangle),np.cos(xangle)],
			]))
		if not freezerot:
			yangle=(mousex-0.5)*np.pi*1.2
		else:
			yangle=0
		rotmat=rotmat.dot(np.array([
			[np.cos(yangle),0.0,-np.sin(yangle)],
			[0.0,1.0,0.0],
			[np.sin(yangle),0.0,np.cos(yangle)],
			]))
		rotmat*=zoom
		nxyz=xyz.dot(rotmat)+[showsz/2,showsz/2,0]
		nxyz_pred=xyz_pred.dot(rotmat)+[showsz/2,showsz/2,0]

		ixyz=nxyz.astype('int32')
		ixyz_pred=nxyz_pred.astype('int32')

		show[:]=background
		dll.render_ball(
			ct.c_int(show.shape[0]),
			ct.c_int(show.shape[1]),
			show.ctypes.data_as(ct.c_void_p),
			ct.c_int(ixyz.shape[0]),
			ixyz.ctypes.data_as(ct.c_void_p),
			c0.ctypes.data_as(ct.c_void_p),
			c1.ctypes.data_as(ct.c_void_p),
			c2.ctypes.data_as(ct.c_void_p),
			ct.c_int(ballradius)
		)
		dll.render_ball(
			ct.c_int(show.shape[0]),
			ct.c_int(show.shape[1]),
			show.ctypes.data_as(ct.c_void_p),
			ct.c_int(ixyz_pred.shape[0]),
			ixyz_pred.ctypes.data_as(ct.c_void_p),
			c0_pred.ctypes.data_as(ct.c_void_p),
			c1_pred.ctypes.data_as(ct.c_void_p),
			c2_pred.ctypes.data_as(ct.c_void_p),
			ct.c_int(ballradius)
		)

	changed=True
	while True:
		if changed:
			render()
			changed=False
		cv2.imshow('show3d',show)
		if waittime==0:
			cmd=cv2.waitKey(10)%256
		else:
			cmd=cv2.waitKey(waittime)%256
		if cmd==ord('q'):
			break
		elif cmd==ord('Q'):
			sys.exit(0)
		if cmd==ord('n'):
			zoom*=1.1
			changed=True
		elif cmd==ord('m'):
			zoom/=1.1
			changed=True
		elif cmd==ord('r'):
			zoom=1.0
			changed=True
		elif cmd==ord('s'):
			break
			# cv2.imwrite('show3d.png',show)
		if waittime!=0:
			break
	return cmd
	
def showpoints(xyz,c0=None,c1=None,c2=None,waittime=0,showrot=False,magnifyBlue=0,freezerot=False,background=(0,0,0),normalizecolor=True,ballradius=10):
	global showsz,mousex,mousey,zoom,changed
	# xyz=xyz-xyz.mean(axis=0)
	radius=((xyz**2).sum(axis=-1)**0.5).max()
	xyz/=(radius*2.2)/showsz
	if c0 is None:
		c0=np.zeros((len(xyz),),dtype='float32')+255
	if c1 is None:
		c1=c0
	if c2 is None:
		c2=c0
	if normalizecolor:
		c0/=(c0.max()+1e-14)/255.0
		c1/=(c1.max()+1e-14)/255.0
		c2/=(c2.max()+1e-14)/255.0
	c0=np.require(c0,'float32','C')
	c1=np.require(c1,'float32','C')
	c2=np.require(c2,'float32','C')

	show=np.zeros((showsz,showsz,3),dtype='uint8')
	def render():
		rotmat=np.eye(3)
		if not freezerot:
			xangle=(mousey-0.5)*np.pi*1.2
		else:
			xangle=0
		rotmat=rotmat.dot(np.array([
			[1.0,0.0,0.0],
			[0.0,np.cos(xangle),-np.sin(xangle)],
			[0.0,np.sin(xangle),np.cos(xangle)],
			]))
		if not freezerot:
			yangle=(mousex-0.5)*np.pi*1.2
		else:
			yangle=0
		rotmat=rotmat.dot(np.array([
			[np.cos(yangle),0.0,-np.sin(yangle)],
			[0.0,1.0,0.0],
			[np.sin(yangle),0.0,np.cos(yangle)],
			]))
		rotmat*=zoom
		nxyz=xyz.dot(rotmat)+[showsz/2,showsz/2,0]

		ixyz=nxyz.astype('int32')
		show[:]=background
		dll.render_ball(
			ct.c_int(show.shape[0]),
			ct.c_int(show.shape[1]),
			show.ctypes.data_as(ct.c_void_p),
			ct.c_int(ixyz.shape[0]),
			ixyz.ctypes.data_as(ct.c_void_p),
			c0.ctypes.data_as(ct.c_void_p),
			c1.ctypes.data_as(ct.c_void_p),
			c2.ctypes.data_as(ct.c_void_p),
			ct.c_int(ballradius)
		)

		if magnifyBlue>0:
			show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=0))
			if magnifyBlue>=2:
				show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=0))
			show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=1))
			if magnifyBlue>=2:
				show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=1))
		if showrot:
			cv2.putText(show,'xangle %d'%(int(xangle/np.pi*180)),(30,showsz-30),0,0.5,cv2.cv.CV_RGB(255,0,0))
			cv2.putText(show,'yangle %d'%(int(yangle/np.pi*180)),(30,showsz-50),0,0.5,cv2.cv.CV_RGB(255,0,0))
			cv2.putText(show,'zoom %d%%'%(int(zoom*100)),(30,showsz-70),0,0.5,cv2.cv.CV_RGB(255,0,0))
	changed=True
	while True:
		if changed:
			render()
			changed=False
		cv2.imshow('show3d',show)
		if waittime==0:
			cmd=cv2.waitKey(10)%256
		else:
			cmd=cv2.waitKey(waittime)%256
		if cmd==ord('q'):
			break
		elif cmd==ord('Q'):
			sys.exit(0)
		if cmd==ord('n'):
			zoom*=1.1
			changed=True
		elif cmd==ord('m'):
			zoom/=1.1
			changed=True
		elif cmd==ord('r'):
			zoom=1.0
			changed=True
		elif cmd==ord('s'):
			cv2.imwrite('show3d.png',show)
		if waittime!=0:
			break
	return cmd
if __name__=='__main__':
	np.random.seed(100)
	showpoints(np.random.randn(1024,3))
