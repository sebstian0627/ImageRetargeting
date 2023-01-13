#VARAIBLES
import numpy as np
import cv2
import maxflow
import matplotlib
import pickle


def generate_weight(img):
  fc,h,w = img.shape
  weights = np.zeros((h,w,4))
  structure = np.zeros((3,3,4))
  
  weights[:,1:-1,0] = np.abs(img[:,:,2:]-img[:,:,:-2]).max(0) #LR
  weights[:,-2,0]=  np.inf
  weights[:,0,0]=  np.inf
  structure[:,:,0]=np.array([[0,0,0],[0,0,1],[0,0,0]])


  weights[0:-1,1:,1] = np.abs(img[:,1:,1:]-img[:,:-1,:-1]).max(0) #LU
  structure[:,:,1]=np.array([[0,0,0],[0,0,0],[0,1,0]])

  weights[1:,1:,2] = np.abs(img[:,1:,0:-1]-img[:,:-1,1:]).max(0)
  structure[:,:,2] = np.array([[0,1,0],[0,0,0],[0,0,0]])

  structure[:,:,3]= np.array([[np.inf,0,0],[np.inf,0,0],[np.inf,0,0]])

  return weights,structure

def remove_seam(img_cube,bgr_img_cube):
  g = maxflow.Graph[float]()
  img = img_cube[0,:,:]
  nodeids = g.add_grid_nodes(img.shape)
#   print(img.shape)
  h,w = img.shape
  weights, structure = generate_weight(img_cube)

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
  fc =img_cube.shape[0]
  mask = (np.arange(w)!= np.vstack(seam)).reshape(1,h,w)
  
  mask = np.repeat(mask, axis=0, repeats=fc)
  mask =  mask==1
  reduced_img = np.reshape(img_cube[mask],(fc,h_t,w_t))
  reduced_bgr_img = np.reshape(bgr_img_cube[mask], (fc,h_t,w_t,3))

  return reduced_img, reduced_bgr_img

def static_video_retargeting(f_name, new_shape):
  # g_img = []
  vcap = cv2.VideoCapture(f_name)

  fc = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
  
  w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(vcap.get(cv2.CAP_PROP_FPS))
  bgr_img_cube = np.zeros((fc,h//2,w//2,3))
  # bgr_img = cv2.imread(f_name)
  # g_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
  # g_img = cv2.resize(g_img, None, fx=0.5, fy=0.5)
  # bgr_img = cv2.resize(bgr_img, None, fx=0.5, fy=0.5)
  # h,w = g_img.shape
  # g_img = cv2.GaussianBlur(g_img, (9,9),0)
  # g_img=g_img/255.0
  i=0
  while vcap.isOpened() and i< fc:
    ret, frame= vcap.read()
    if(not ret):
      break
    bgr_img_cube[i,:,:] = cv2.resize(frame, None, fx=0.5, fy=0.5)
    del frame
    i+=1

  g_img_cube = np.sum(bgr_img_cube, axis=3)/3

  # reduce



  h_d, w_d = np.ceil(new_shape[0]*h)//2, np.ceil(w*new_shape[1])//2
  # print(h,w,h_d,w_d)
  delete_rows_h = h//2-h_d
  delete_cols_w = w//2-w_d

  num_seams_y =int( delete_rows_h)
  num_seams_x =int( delete_cols_w)

  for i in range(num_seams_x):
    print(i, 'x')
    g_img_cube, bgr_img_cube = remove_seam(g_img_cube, bgr_img_cube)

  # g_img = cv2.rotate(g_img, cv2.cv2.ROTATE_90_CLOCKWISE)
  # bgr_img = cv2.rotate(bgr_img, cv2.cv2.ROTATE_90_CLOCKWISE)
  g_cube_1 = []
  bgr_vid_cube_1=[]
  for i in range(fc):
    g_cube_1.append(np.rot90(g_img_cube[i,:,:]))
    bgr_vid_cube_1.append(np.rot90(bgr_img_cube[i,:,:,:]))
  print(g_cube_1[0].shape, 'shape')

  g_img_cube=np.array(g_cube_1)
  bgr_img_cube = np.array(bgr_vid_cube_1)

  for i in range(num_seams_y):
    print(i)
    g_img_cube, bgr_img_cube = remove_seam(g_img_cube, bgr_img_cube)
  
  g_cube_1 = []
  bgr_vid_cube_1=[]
  for i in range(fc):
    g_cube_1.append(np.rot90(g_img_cube[i,:,:], k=-1))
    bgr_vid_cube_1.append(np.rot90(bgr_img_cube[i,:,:,:], k=-1))
  g_img_cube=np.array(g_cube_1)
  bgr_img_cube = np.array(bgr_vid_cube_1)

  return g_img_cube, bgr_img_cube


# img = cv2.imread('/content/drive/MyDrive/Project_dataset/Project_dataset/Image_retargeting/Taks_1_dataset/1.jpg')
# g_img, bgr_img = image_retargeting('/content/drive/MyDrive/Project_dataset/Project_dataset/Image_retargeting/Taks_1_dataset/1.jpg',(1,0.75),image=True)
# rgb_img_orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
# rgb_img_test = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) 
# #cv2.waitKey(0)

gvc = static_video_retargeting('ratatouille1-resize.mov', (0.5,0.75))
pickle.dump(gvc, open('static_rat_first_half.pkl', 'wb'))