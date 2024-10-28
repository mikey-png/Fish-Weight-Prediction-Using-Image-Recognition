import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
color_image = cv2.imread('./img_test/color_24.tiff')  # 颜色图像
depth_image_8 = cv2.imread('./img_test/depth_8_debug.tiff', cv2.IMREAD_GRAYSCALE)  # 8位深度图
depth_image_16 = cv2.imread('./img_test/depth_16.tiff', cv2.IMREAD_UNCHANGED)  # 16位深度图（深度信息较准确）
depth_image_colored = cv2.imread('./img_test/depth_colored_24.tiff')  # 彩色深度图，用于可视化

plt.figure(figsize=(10, 8))
plt.imshow(depth_image_8, cmap='jet')
plt.title('Depth Information Visualization')
plt.colorbar(label='Depth Value')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 8))
plt.imshow(depth_image_16, cmap='jet')
plt.title('Depth Information Visualization')
plt.colorbar(label='Depth Value')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 8))
plt.imshow(depth_image_colored, cmap='jet')
plt.title('Depth Information Visualization')
plt.colorbar(label='Depth Value')
plt.axis('off')
plt.show()