import cv2
import numpy as np
import os
import maxflow
import matplotlib.pyplot as plt


#VARAIBLES
def generate_weight(img):
  h,w = img.shape
  weights = np.zeros((h,w,4))
  structure = np.zeros((3,3,4))
  
  weights[:,1:-1,0] = np.abs(img[:,2:]-img[:,:-2]) #LR
  weights[:,-2,0]=  np.inf
  weights[:,0,0]=  np.inf
  structure[:,:,0]=np.array([[0,0,0],[0,0,1],[0,0,0]])


  weights[0:-1,1:,1] = np.abs(img[1:,1:]-img[:-1,:-1]) #LU
  structure[:,:,1]=np.array([[0,0,0],[0,0,0],[0,1,0]])

  weights[1:,1:,2] = np.abs(img[1:,0:-1]-img[:-1,1:])
  structure[:,:,2] = np.array([[0,1,0],[0,0,0],[0,0,0]])

  structure[:,:,3]= np.array([[np.inf,0,0],[np.inf,0,0],[np.inf,0,0]])

  return weights,structure

def coords2id(list_of_coords,img_shape):
  h,w = img_shape
  list_of_ids=[]

  for i in list_of_coords:
    id = i[0]*w + i[1]
    list_of_ids.append(id)

  return np.array(list_of_ids)

def ids2coord(list_of_ids, img_shape):
  h,w = img_shape
  list_of_coords = []
  for i in list_of_ids:
    x = int(i/w)
    y = i%w
    list_of_coords.append([x,y])
  list_of_coords = np.array(list_of_coords)
  return list_of_coords


def remove_seam(img,bgr_img):
  g = maxflow.Graph[float]()
  nodeids = g.add_grid_nodes(img.shape)
  
  h,w = img.shape
  weights, structure = generate_weight(img)

  g.add_grid_edges(nodeids, structure = structure[:,:,3], symmetric=False)
  for i in range(3):
    g.add_grid_edges(nodeids, structure= structure[:,:,i], weights = weights[:,:,i], symmetric=False)


  left_most = np.concatenate((np.arange(img.shape[0]).reshape(1, img.shape[0]), np.zeros((1, img.shape[0])))).astype(np.uint64)
  left_most = np.ravel_multi_index(left_most,img.shape)
  g.add_grid_tedges(left_most, np.inf, 0)

  right_most = np.concatenate((np.arange(img.shape[0]).reshape(1, img.shape[0]), np.ones((1, img.shape[0])) * (np.size(img, 1) - 1))).astype(np.uint64)
  right_most = np.ravel_multi_index(right_most, img.shape)
  g.add_grid_tedges(right_most, 0, np.inf)

  g.maxflow()

  seam = g.get_grid_segments(nodeids)
  seam = np.sum((seam==0),axis=1) -1
  seam = seam.reshape(h,1)

  #apply seam carving

  h_t, w_t = h,w-1

  mask = np.arange(w)!= np.vstack(seam)
  reduced_img = np.reshape(img[mask],(h_t,w_t))
  reduced_bgr_img = np.reshape(bgr_img[mask], (h_t,w_t,3))

  return reduced_img, reduced_bgr_img

def image_retargeting(f_name, new_shape, img= None, image=False):
  g_img = []
  if not image:
    g_img = img
  else:
    bgr_img = cv2.imread(f_name)
    g_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
  # g_img = cv2.resize(g_img, None, fx=0.5, fy=0.5)
  # bgr_img = cv2.resize(bgr_img, None, fx=0.5, fy=0.5)
  h,w = g_img.shape
  # g_img = cv2.GaussianBlur(g_img, (9,9),0) # for testing of image 2
  g_img=g_img/255.0

  h_d, w_d = np.ceil(new_shape[0]*h), np.ceil(w*new_shape[1])
  print(h,w,h_d,w_d)
  delete_rows_h = h-h_d
  delete_cols_w = w-w_d

  num_seams_y =int( delete_rows_h)
  num_seams_x =int( delete_cols_w)

  for i in range(num_seams_x):
    print(i)
    g_img, bgr_img = remove_seam(g_img, bgr_img)

  g_img = cv2.rotate(g_img, cv2.cv2.ROTATE_90_CLOCKWISE)
  bgr_img = cv2.rotate(bgr_img, cv2.cv2.ROTATE_90_CLOCKWISE)
  
  for i in range(num_seams_y):
    print(i)
    g_img, bgr_img = remove_seam(g_img, bgr_img)
  
  g_img = cv2.rotate(g_img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
  bgr_img = cv2.rotate(bgr_img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

  return g_img, bgr_img


img = cv2.imread('/content/drive/MyDrive/Project_dataset/Project_dataset/Image_retargeting/Taks_1_dataset/1.jpg')
g_img, bgr_img = image_retargeting('/content/drive/MyDrive/Project_dataset/Project_dataset/Image_retargeting/Taks_1_dataset/1.jpg',(1,0.75),image=True)
rgb_img_orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
rgb_img_test = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)