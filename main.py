import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('demo.jpeg')  # Replace with your image file

# Manual grayscale conversion using weighted sum
manual_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)

# OpenCV grayscale conversion
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Noise reduction
gaussian_blur = cv2.GaussianBlur(gray_img, (5,5), 0)
median_blur = cv2.medianBlur(gray_img, 5)

# Edge detection
edges_canny = cv2.Canny(gaussian_blur, 50, 150)
sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
sobel = cv2.magnitude(sobelx, sobely)
sobel = cv2.convertScaleAbs(sobel)
laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# Histogram Equalization
equalized_img = cv2.equalizeHist(gray_img)

# Performance metrics (edge pixel counts)
canny_edges_count = np.sum(edges_canny > 0)
sobel_edges_count = np.sum(sobel > 0)
laplacian_edges_count = np.sum(laplacian > 0)

print(f"Canny edge pixels: {canny_edges_count}")
print(f"Sobel edge pixels: {sobel_edges_count}")
print(f"Laplacian edge pixels: {laplacian_edges_count}")

# Plot histograms
plt.figure(figsize=(12,6))

plt.subplot(2,3,1)
plt.title('Original Grayscale Histogram')
plt.hist(gray_img.ravel(), 256, [0,256])

plt.subplot(2,3,2)
plt.title('Equalized Histogram')
plt.hist(equalized_img.ravel(), 256, [0,256])

plt.subplot(2,3,3)
plt.title('Canny Edges Histogram')
plt.hist(edges_canny.ravel(), 256, [0,256])

plt.subplot(2,3,4)
plt.title('Sobel Edges Histogram')
plt.hist(sobel.ravel(), 256, [0,256])

plt.subplot(2,3,5)
plt.title('Laplacian Edges Histogram')
plt.hist(laplacian.ravel(), 256, [0,256])

plt.tight_layout()
plt.show()

# Show images
cv2.imshow('Original Image', img)
cv2.imshow('Manual Grayscale', manual_gray)
cv2.imshow('OpenCV Grayscale', gray_img)
cv2.imshow('Equalized Grayscale', equalized_img)
cv2.imshow('Gaussian Blur', gaussian_blur)
cv2.imshow('Median Blur', median_blur)
cv2.imshow('Canny Edges', edges_canny)
cv2.imshow('Sobel Edges', sobel)
cv2.imshow('Laplacian Edges', laplacian)

cv2.waitKey(0)
cv2.destroyAllWindows()
