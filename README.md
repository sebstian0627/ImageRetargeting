# Image Retargeting

## Requirements
- PyMaxflow
- opencv-python
- numpy
- Matplotlib
- ffmpeg

##Image Retargeting

A BGR image is passed in the main function. The graph is constructed using the Grayscale version of the image. The weights are added according to the Forward Energy Function. Pymaxflow is used to find the vertical seam from the image. After the cut is obtained, the seam is removed from the image. We iteratively repeat this process to reduce the width one by one. To reduce the width of the image, we rotate the image and remove the seams again. and rotate it back.

- generate_weights - takes the gray image and returns a weight and structure array to add grid edges in the graph.
- coords2ids - takes into list of (x,y) coordinates and shape of the gray image to generate ids of the graph, works as same as ravel multi index function of numpy.
- ids2coord -  takes into a list of ids and image shape and returns the (x,y) coords
- remove seam - takes into the gray image and the bgr image, generates a graph using PyMaxflow, find the arc weights using the forward energy cost function, adds terminal edges(arcs to S and T), and computes the max flow or min cut, once the seam is found, we remove it from the image with the help of the generated mask. and the mask is applied to both gray and bgr image and returned as output.
- image retargeting - takes into input either a image path or the image object itself aling with the target scale and reduces the dimensions of the image accordingly.


## Video retargeting

Three techniques are implemented in the project for this:

- Frame wise Image retargeting - The image retargeting code is applied to each image frame and is returned as the output
  * Functions of this pipeline are similar to the Image retargeting
- Static Seams for the whole video - A single seam, independent of time is calculated using the similar implementation of the graph cut for image retargeting method using forward energy cost function.
  * generate_weights -  take into input the img cube and finds the weights of the graph using the forward energy cost.
  * remove_seam - takes into input the bgr image cube and the gray scale img cube to return the reduced image. A graph is constructed using the gray img cube, the dimensions of graph grid is h x w, where h is height of the img frame and w is the width. Using the generate_weight function weights are added and maxflow is computed, this results in 1-D seam to be generated, and a mask is generated with the help of the image and this mask is applied to each and every frame of the frame in the video cube.
  * static_video_retargeting - This is the main function, takes in the video path, reads each frame of the video, append it to make the cube, and gray scale cube is generated by averaging at the channel axis, Then seams are removed according to the scale provided in the code
- Using Video cube to find 2D seam manifolds - A video is passed as an input, using which a frame/video cube is created, the dimensions of the cube are frame_count x h x w x 3, then a grayscale cube is computed taking the average accross the color channel axis. Then using the gray scale cube the graph is constructed and arc weights are added. with the help of Pymaxflow, we find the min cut, and the 2d vertical seam of the cube. We shift the seam by 1 pixel, and take the Xor to find the mask of cube(it is 0 at seam, and 1 otherwise). Using this seam, the retargeted video cube is constructed. We repeat this iteratively till we reach the required dimensions of the cube. For the horizontal seams, we rotate the image by 90 and apply the same method, then we rotate it back by -90 to obtain the frames. Then we use ffmpeg to convert the image frames into the video. The running time for this code is bit too much, so I have computed the results only with 50% of the frames with resizing input by 0.5.
