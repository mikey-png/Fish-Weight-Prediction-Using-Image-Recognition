# Fish-Weight-Prediction-Using-Image-Recognition
This code processes fish images by isolating the fish using HSV masking and contour detection. It calculates the fish's length, height, and area, then saves these measurements along with the fish ID and weight to an Excel file. Processed images with bounding boxes are also saved for reference.

分别提取每个图片中鱼的主体
<div align="center">
    <img src="https://github.com/user-attachments/assets/3f6b5daf-f816-4700-83e1-e334483389d7" alt="image" />
</div>

计算长度、宽度、像素面积 与原始数据中的重量，组合为新的训练样本 
<div align="center">
    <img src="https://github.com/user-attachments/assets/89b8f7bb-3370-49c0-9086-1470791fdb38" alt="image" />
</div>

训练样本中可见像素面积和重量有一定的线性相关性
<div align="center">
    <img src="https://github.com/user-attachments/assets/a2cdf8f6-b961-4902-b1e5-cc46b814a7aa" alt="image" />
</div>

使用RF+SVM训练 在测试集可达 90%以上准确率
<div align="center">
    <img src="https://github.com/user-attachments/assets/2aaa4d73-7477-48d0-817c-57c1637873ad" alt="image" />
</div>

Datasource：https://www.kaggle.com/datasets/miokee/fish-img
