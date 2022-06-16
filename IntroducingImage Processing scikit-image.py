Introducing Image Processing and scikit-image
#purpose#
#1.visualization:objects are not visible
#2.For a better image:image sharpening and restoration 
#3.image retrieval:seek for image of interest
#4.image recognition

from skimage import data
rocket_image = data.rocket() #从数据中调取rocket的图像
#use the function .shape() from NumPy, to obtain the image shape (Height, Width, Dimensions)
 
#二维彩色图像以RGB-3层中的二维数组 channels：red/green/blue
#Grayscaled images:数据代表灰度强度，储存为8位整数.Do not have any color information.Only one color channel.
#RGB<->Grayscaled
from skimage import color
grayscale = color.rgb2gray(original)
rgb = color.gray2rgb(grayscale)

#show_image function来预加载函数，使用Matplotlib显示图像#
def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
plt.show()

from skimage import color
grayscale = color.rgb2gray(original)
show_image(grayscale, "Grayscale")

##Numpy for Image##
#Fundamental processing:flipping,extract and anlayze features,
#Loading the image using Matplotlib
madrid_image = plt.imread('/madrid.jpeg')
type(madrid_image)
#return:<class 'numpy.ndarray'>

#Color with NumPy#对颜色进行切片处理 
# Obtaining the red values of the image
red = image[:, :, 0]    #Given the height,width pixels,and select the only values of the first color layer
# Obtaining the green values of the image
green = image[:, :, 1]
# Obtaining the blue values of the image
blue = image[:, :, 2]
#Display with gray colormap
plt.imshow(red, cmap="gray")
plt.title('Red'/'Green'/'Blue')
plt.axis('off')
plt.show() #we can see the different density of images
# Accessing the shape of the image
madrid_image.shape #return:(426, 640, 3)
# Accessing the shape of the image
madrid_image.size #return:817920

# Flip the image in up direction
vertically_flipped = np.flipud(madrid_image) #ud->up/down
show_image(vertically_flipped, 'Vertically flipped image')
# Flip the image in left direction
horizontally_flipped = np.fliplr(madrid_image) #lr->lift/right
show_image(horizontally_flipped, 'Horizontally flipped image') #(data input,'title')

#historgrams#
#Origin image->Red/Green/Blue Historgram
#Applications:analysis,thresholding,brightness and contrast,equalize an image
# Red color of the image,matplotlib
red = image[:, :, 0]
# Obtain the red histogram
plt.hist(red.ravel(), bins=256) #ravel() to make these color values an array of one flat dimension.
#Visualizing blue histrogram
blue = image[:, :, 2]
plt.hist(blue.ravel(), bins=256)
plt.title('Blue Histogram')
plt.show()

##Getting start with Thresholding##
#Thresholding#
#Definiton:partition background and foreground,by making it black and white->simplest method of image segmentation
#EG:
# 255(white) if pixel>thresh value
# 0(black) if pixel<thresh value
#So we must cinvert them into grayscale image

# Obtain the optimal threshold value
thresh = 127
# Apply thresholding to the image
binary = image > thresh
# Show the original and thresholded
show_image(image, 'Original')
show_image(binary, 'Thresholded')

#Inverted Thresholding#和pervious one刚好黑白颠倒
# Obtain the optimal threshold value
thresh = 127
# Apply thresholding to the image
inverted_binary = image <= thresh
# Show the original and thresholded
show_image(image, 'Original')
show_image(inverted_binary,'Inverted thresholded')

#Categories#
#Global or histogram based:good for uniform background
#local or adaptive:uneven background
#Try more thresholding algorithms#
from skimage.filters import try_all_threshold
# Obtain all the resulting images
fig, ax = try_all_threshold(image, verbose=False)
# Showing resulting plots
show_plot(fig, ax)

#optimal thresh value-global uniform background#
# Import the otsu threshold function
from skimage.filters import threshold_otsu
# Obtain the optimal threshold value
thresh = threshold_otsu(image)
# Apply thresholding to the image
binary_global = image > thresh
# Show the original and binarized image
show_image(image, 'Original')
show_image(binary_global, 'Global thresholding')

#optimal thresh value-local uneven background#
# Import the local threshold function
from skimage.filters import threshold_local
# Set the block size to 35,surround each pixel,known as local neighborhoods
block_size = 35 
# Obtain the optimal local thresholding,a constant subtracted from the mean of blocks to calculate the local threshold value
local_thresh = threshold_local(text_image, block_size, offset=10)
# Apply local thresholding and obtain the binary image
binary_local = text_image > local_thresh
# Show the original and binarized image
show_image(text_image, 'Original')
show_image(binary_local, 'Local thresholding')