import math
import cv2

# load the original and generated images
original_image = cv2.imread('rgb/1341847980.722988.png')
one_FPS_image = cv2.imread('results/1_FPS_30000_iter/pose_1.png')
zero_point_two_FPS_image = cv2.imread('results/0.2_FPS_30000_iter/pose_1.png')
zero_point_one_FPS_image = cv2.imread('results/0.1_FPS_60000_iter/pose_1.png')

one_FPS_image = cv2.resize(one_FPS_image, (640, 480))
zero_point_two_FPS_image = cv2.resize(zero_point_two_FPS_image, (640, 480))
zero_point_one_FPS_image = cv2.resize(zero_point_one_FPS_image, (640, 480))

# convert the images to grayscale
original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
one_FPS_image = cv2.cvtColor(one_FPS_image, cv2.COLOR_BGR2GRAY)
zero_point_two_FPS_image = cv2.cvtColor(zero_point_two_FPS_image, cv2.COLOR_BGR2GRAY)
zero_point_one_FPS_image = cv2.cvtColor(zero_point_one_FPS_image, cv2.COLOR_BGR2GRAY)

for generated_image_gray in (one_FPS_image, zero_point_two_FPS_image, zero_point_one_FPS_image):
    # calculate the mean squared error (MSE)
    mse = ((original_image_gray - generated_image_gray) ** 2).mean()

    # calculate the maximum pixel value
    max_pixel_value = 255.0

    # calculate the PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))

    print("PSNR:", psnr)