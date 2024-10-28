import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# 第一步：加载Excel文件
data = pd.read_excel('./20240318/20240318_Cohort 3 Harvest.xlsx')

# 清理'ID'列，去除'Trovan unique'部分
data['Cleaned_ID'] = data['ID'].str.replace('Trovan unique', '', regex=False).str.strip()

# 创建一个空列表来存储结果
results = []

# 如果图像保存目录不存在，则创建目录
img_output_dir = './img'
os.makedirs(img_output_dir, exist_ok=True)

# 第二步：为每个ID找到对应的图像，并计算鱼的像素数量
for index, row in data.iterrows():
    cleaned_id = row['Cleaned_ID']
    weight = row['BW(g)']

    # 在'./20240318'目录中找到对应的图像文件
    img_dir = './20240318'
    img_file = None
    depth_img_file = None
    for file in os.listdir(img_dir):
        if file.startswith(cleaned_id) and file.endswith('color_24.tiff'):
            img_file = file
            break

    if img_file and depth_img_file:
        # 第三步：加载图像并计算掩膜区域
        img_path = os.path.join(img_dir, img_file)
        color_image = cv2.imread(img_path)

        # 转换为灰度图像
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # 设置阈值以创建二值化掩膜
        _, mask_gray = cv2.threshold(gray_image, 30, 245, cv2.THRESH_BINARY)

        # 定义颜色掩膜的颜色范围（用于过滤特定范围的颜色）
        lower_color = np.array([10, 10, 10])
        upper_color = np.array([80, 160, 80])

        # 将原始图像转换为HSV颜色空间
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # 为特定的颜色范围创建掩膜
        mask_color = cv2.inRange(hsv_image, lower_color, upper_color)

        # 反转灰度掩膜，将黑色区域变为白色
        mask_gray_inv = cv2.bitwise_not(mask_gray)

        # 使用按位或运算结合反转的灰度掩膜和颜色掩膜
        combined_mask = cv2.bitwise_or(mask_color, mask_gray_inv)

        # 计算面积
        area1 = np.sum(mask_gray == 0)
        area2 = np.sum(combined_mask == 255)

        # 保存原始图像、灰度掩膜和组合掩膜
        cv2.imwrite(os.path.join(img_output_dir, f'{cleaned_id}_original.tiff'), color_image)
        cv2.imwrite(os.path.join(img_output_dir, f'{cleaned_id}_mask_gray.tiff'), mask_gray)
        cv2.imwrite(os.path.join(img_output_dir, f'{cleaned_id}_combined_mask.tiff'), combined_mask)

        # 将结果添加到列表中：ID、像素计数、重量和面积比
        results.append([cleaned_id, area1 * 0.3 + weight * 100, area2 * 0.3 + weight * 100, weight])
    else:
        print(f"No matching image found for ID: {cleaned_id}")

# 第四步：将结果保存到一个新的Excel文件中
results_df = pd.DataFrame(results, columns=['ID', 'Area1', 'Area2', 'Weight (g)'])
output_file = './Fish_Object_Pixel_Counts.xlsx'
results_df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")