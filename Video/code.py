import numpy as np
from random import random
import maxflow
import cv2
import pickle

 

#video cube is created of dimenion framecount,h,w

def generate_weights(g_cube, type):
  #UD edges
  if type == 'UD':
    fc,h,w = g_cube.shape
    structure = np.zeros((3,3,3))
    weights = np.zeros((fc,h,w))
    structure[1,2,1]=1
    weights[:,:-1,1:]= np.abs(g_cube[:,:-1,1:]-g_cube[:,1:,:-1])
    return structure, weights

  if type == 'DU':
    fc,h,w = g_cube.shape
    structure = np.zeros((3,3,3))
    weights = np.zeros((fc,h,w))
    structure[1,0,1] = 1
    weights[:,1:,1:] = np.abs(g_cube[:,1:,1:] -g_cube[:,:-1,:-1])
    return structure, weights


  if type == 'LR':
    fc,h,w = g_cube.shape
    structure = np.zeros((3,3,3))
    weights = np.zeros((fc,h,w))
    structure[1,1,2] =1
    weights[:,:,1:-1] = np.abs(g_cube[:,:,2:]-g_cube[:,:,:-2])
    weights[:,:,0] =np.inf
    weights[:,:,-2] =np.inf
    return structure, weights

  if type =='BF':
    fc,h,w = g_cube.shape
    structure = np.zeros((3,3,3))
    weights = np.zeros((fc,h,w))
    structure[2,1,1] = 1
    weights[0:-1,:,1:] = np.abs(g_cube[:-1,:,1:] - g_cube[1:,:,:-1])
    return structure, weights

  if type == 'FB':
    fc,h,w = g_cube.shape
    structure = np.zeros((3,3,3))
    weights = np.zeros((fc,h,w))
    structure[0,1,1]=1
    weights[0:-1,:,1:] = np.abs(g_cube[0:-1,:,0:-1]- g_cube[1:,:,1:])
    return structure, weights


def video_retarget( g_cube):
  g = maxflow.Graph[float]()
  node_ids = g.add_grid_nodes(g_cube.shape)
  
  structure, weights = generate_weights(g_cube , type ='LR' )
  g.add_grid_edges(node_ids, structure = structure, symmetric = False, weights=weights)
  
  structure, weights = generate_weights(g_cube,  type ='UD'  )
  g.add_grid_edges(node_ids, structure = structure, symmetric = False, weights=weights)
  
  structure, weights = generate_weights(g_cube, type= 'DU'  )
  g.add_grid_edges(node_ids, structure = structure, symmetric = False, weights=weights)
  
  structure = np.zeros((3,3,3))
  structure[1,:,0]=np.inf
  structure[2,1,0]=np.inf
  structure[0,1,0]=np.inf
  g.add_grid_edges(node_ids, structure=structure)
  
  structure, weights = generate_weights(g_cube , type ='BF' )
  g.add_grid_edges(node_ids, structure = structure, symmetric = False, weights=weights)
  
  structure, weights = generate_weights(g_cube, type ='FB'  )
  g.add_grid_edges(node_ids, structure = structure, symmetric = False, weights=weights)
  
  structure=[]
  weights=[]
  g.add_grid_tedges(node_ids[:,:,0], np.inf, 0)
  g.add_grid_tedges(node_ids[:,:,-1], 0, np.inf)
  
  g.maxflow()

  seam2d = g.get_grid_segments(node_ids)
  
  g=[]
  del g

  return seam2d

def reshaping(seam2d, g_cube):
  fc,h,w = g_cube.shape
  Shifted_seam = np.ones((fc, h,w))
  Shifted_seam[:,:,:-1] = seam2d[:,:,1:] 
  
  mask = 1 - np.logical_xor(Shifted_seam, seam2d)
  del Shifted_seam

  w_d = w-1
  
  g_cube = g_cube[mask==1].reshape(fc,h,w_d)
  
  return g_cube, mask


def main(f_name):
  vcap = cv2.VideoCapture(f_name)
  fc = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
  fc =fc//2
  w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH ))
  h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
  fps = int(vcap.get(cv2.CAP_PROP_FPS))
  print(h,w,fc,fps)
  bgr_video_cube = np.zeros((fc,h//2,w//2,3))
  count = 0
  while vcap.isOpened() and i < fc:
    ret, frame = vcap.read()
    count+=1
    if (not ret):
      break
    # bgr_video_cube[i]=frame
    bgr_video_cube[i]=cv2.resize(frame, None, fx=0.5, fy=0.5)
    del frame
    i+=1
  g_vid_cube = np.sum(bgr_video_cube, axis=3)/3
  reduced_width = int(np.ceil(0.375*w))
  for iter1 in range(w//2-reduced_width):

    seam2d = video_retarget( g_vid_cube)
    gray_vid_cube, mask = reshaping(seam2d, g_vid_cube)
    g_vid_cube = np.array(gray_vid_cube)
    fc, h1,w1 = g_vid_cube.shape
    bgr_video_cube = bgr_video_cube[mask==1].reshape(fc,h1,w1,3)
    print(iter1,'w')

  g_cube_1 = []
  bgr_vid_cube1 = []
  for i in range(fc):
    g_cube_1.append(np.rot90(g_vid_cube[i,:,:]))
    bgr_vid_cube1.append(np.rot90(bgr_video_cube[i,:,:,:]))
  print(g_cube_1[0].shape)
  g_vid_cube = np.array(g_cube_1)
  bgr_video_cube = np.array(bgr_vid_cube1)

  for iter1 in range(int(h//2-np.ceil(0.25*h))):

    seam2d = video_retarget( g_vid_cube)
    gray_vid_cube, mask = reshaping(seam2d, g_vid_cube)
    g_vid_cube = np.array(gray_vid_cube)
    fc, h1,w1 = g_vid_cube.shape
    bgr_video_cube = bgr_video_cube[mask==1].reshape(fc,h1,w1,3)

    print(iter1,'h')

  g_cube_1 = []
  bgr_vid_cube1 = []
  for i in range(fc):
    g_cube_1.append(np.rot90(g_vid_cube[i,:,:],-1))
    bgr_vid_cube1.append(np.rot90(bgr_video_cube[i,:,:,:],-1))
  g_vid_cube = np.array(g_cube_1)
  bgr_video_cube = np.array(bgr_vid_cube1)

  vcap.release()
  return gray_vid_cube, bgr_video_cube






# gvc = main('golf.mov')
# pickle.dump(gvc, open('h_golf_cube_2.pkl','wb'))

gvc = main('ratatouille1-resize.mov')
pickle.dump(gvc, open('h_rat_cube_2.pkl','wb'))