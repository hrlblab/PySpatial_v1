import cv2
import numpy as np

def reduce_boundary(mask,wsi_img):
    # 查找连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 初始化最小和最大坐标
    x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf

    # 遍历每个连通组件，更新最小和最大坐标
    for stat in stats[1:]:  # 跳过背景
        x, y, w, h = stat[:4]
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # 裁剪mask中对象部分
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    cropped_wsi=wsi_img[y_min:y_max, x_min:x_max]

   # print(f"Bounding box coordinates: x={x_min}, y={y_min}, width={x_max - x_min}, height={y_max - y_min}")
    return cropped_mask, cropped_wsi,(x_min, y_min, x_max, y_max)

