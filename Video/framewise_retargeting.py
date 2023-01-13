import numpy as np
import cv2
# dataset_Path = '/content/drive/My_Drive/Project_dataset/Project_dataset/'
import maxflow
import matplotlib.pyplot as plt
import pickle

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

  # weights[:,:,3] = np.inf
  # structure[:,:,3] = np.array([[1,0,0],[0,0,0],[0,0,0]])

  # weights[:,:,3] = np.inf
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
  node_inf = np.inf
  i_mult = 1

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
    # print('I entered this if ele')
    bgr_img = img
    g_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
  else:
    bgr_img = cv2.imread(f_name)
    g_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
  # g_img = cv2.resize(g_img, None, fx=0.5, fy=0.5)
  # bgr_img = cv2.resize(bgr_img, None, fx=0.5, fy=0.5)
  h,w = g_img.shape

  g_img=g_img/255.0

  h_d, w_d = np.ceil(new_shape[0]*h), np.ceil(w*new_shape[1])
#   print(h,w,h_d,w_d)
  delete_rows_h = h-h_d
  delete_cols_w = w-w_d

  num_seams_y =int( delete_rows_h)
  num_seams_x =int( delete_cols_w)

  for i in range(num_seams_x):
    # print(i)
    g_img, bgr_img = remove_seam(g_img, bgr_img)

  g_img = cv2.rotate(g_img, cv2.cv2.ROTATE_90_CLOCKWISE)
  bgr_img = cv2.rotate(bgr_img, cv2.cv2.ROTATE_90_CLOCKWISE)
  
  for i in range(num_seams_y):
    # print(i)
    g_img, bgr_img = remove_seam(g_img, bgr_img)
  
  g_img = cv2.rotate(g_img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
  bgr_img = cv2.rotate(bgr_img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

  return g_img, bgr_img



def part1(f_name):
  vcap = cv2.VideoCapture(f_name)
  fc = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
  w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH ))
  h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
  fps = int(vcap.get(cv2.CAP_PROP_FPS))
  print(fps, w,h,fc)
  frames_g=[]
  frames_bgr=[]
  while vcap.isOpened() and i< fc:
    print(i, '\t')
    ret, frame = vcap.read()
    if (not ret):
      break
    frame =  cv2.resize(frame, None, fx=0.5, fy=0.5)
    g_img, bgr_img = image_retargeting('',(0.5,0.75), img=frame, image=False)
    # vw.write(bgr_img)
    i+=1
    frames_g.append(g_img)
    frames_bgr.append(bgr_img)
  pickle.dump((frames_g, frames_bgr),open('img_ret_rat_frames.pkl','wb'))

  vcap.release()
part1('ratatouille1-resize.mov')