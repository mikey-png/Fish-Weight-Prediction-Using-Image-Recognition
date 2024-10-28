import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

# 设置matplotlib支持中文的字体
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取Excel数据
data = pd.read_excel('./datanew/20240319/20240319_Cohort 3 Harvest.xlsx')

# 创建一个列表用于存储结果
results = []

# 指定目录路径
dir_path = './datanew/20240319/'

# 创建保存转换后图像的目录
output_dir = './datanew/20240319转换/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历Excel中的每一行
for index, row in data.iterrows():
    ID = row['ID']
    Weight = row['BW']
    ID_clean = ID.replace("Trovan unique", "").strip()  # 去除“Trovan unique”部分

    # 查找符合条件的文件
    image_path = None
    for filename in os.listdir(dir_path):
        if filename.startswith(ID_clean) and filename.endswith('color_24.tiff'):
            image_path = os.path.join(dir_path, filename)
            break  # 找到第一个符合条件的文件后停止查找

    # 检查是否找到符合条件的文件
    if image_path:
        print(f"找到的文件路径是: {image_path}")
    else:
        print(f"没有找到以 {ID_clean} 开头并且以 color_24.tiff 结尾的文件。")
        # 如果没有找到图像，跳过此循环
        continue

    # 读取图像
    image = cv2.imread(image_path)

    # 将BGR格式转换为HSV（色调-饱和度-明度）
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 蓝色背景的掩膜
    lower_blue = np.array([90, 50, 50])  # 蓝色的最低HSV值
    upper_blue = np.array([130, 255, 255])  # 蓝色的最高HSV值
    mask = cv2.inRange(hsv, lower_blue, upper_blue)  # 保留蓝色部分，去除其他颜色
    # 蓝色区域（背景）变为白色（255），其他部分（鱼和图像边缘区域）变为黑色（0）

    # 反转蓝色区域的掩膜：蓝色区域变为黑色，其他部分变为白色
    result_inverted = cv2.bitwise_not(mask)

    # 去除图像边缘的白色区域
    # 创建一个与原图像大小相同的黑色背景
    black_background = np.zeros_like(image)
    # 获取图像的尺寸
    height, width = result_inverted.shape

    # 定义中心区域
    center_height = int(height * 0.75)
    center_width = int(width * 0.75)

    start_y = (height - center_height) // 2
    end_y = start_y + center_height
    start_x = (width - center_width) // 2
    end_x = start_x + center_width

    # 裁剪中心区域
    cropped_result = result_inverted[start_y:end_y, start_x:end_x]

    # 将裁剪后的区域粘贴到新背景上
    black_background[start_y:end_y, start_x:end_x] = cv2.merge([cropped_result, cropped_result, cropped_result])

    # 查找最小的包围矩形
    contours, _ = cv2.findContours(cropped_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 选择最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # 画出最小的包围矩形
        cv2.rectangle(black_background[start_y:end_y, start_x:end_x], (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 获取鱼的长度和高度
        fish_length = h
        fish_height = w
        # 计算鱼的面积
        fish_area = cv2.contourArea(largest_contour)
    else:
        print(f"未找到鱼的轮廓，跳过ID: {ID_clean}")
        continue

    # 保存结果到列表
    results.append({
        'ID': ID_clean,
        'fish_length': fish_length,
        'fish_height': fish_height,
        'fish_area': fish_area,
        'Weight': Weight
    })

    # 保存带有包围框的图像
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(black_background, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'ID: {ID_clean}')
    plt.savefig(f'{output_dir}{ID_clean}.png')
    plt.close()

# 将结果保存到Excel
results_df = pd.DataFrame(results)
results_df.to_excel(f'{output_dir}fish_measurements-{20240319}.xlsx', index=False)
print("所有处理完成，结果已保存到 fish_measurements.xlsx")
