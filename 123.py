import numpy as np
from matplotlib import pyplot as plt

def erosion(image, kernel):
    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape
    output_image = np.zeros((img_h, img_w), dtype=np.uint8)
    offset_h = kernel_h // 2
    offset_w = kernel_w // 2

    for i in range(offset_h, img_h - offset_h):
        for j in range(offset_w, img_w - offset_w):
            region = image[i - offset_h:i + offset_h + 1, j - offset_w:j + offset_w + 1]
            if np.all(region == kernel):
                output_image[i, j] = 255
            else:
                output_image[i, j] = 0
    return output_image

def dilation(image, kernel):
    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape
    output_image = np.zeros((img_h, img_w), dtype=np.uint8)
    offset_h = kernel_h // 2
    offset_w = kernel_w // 2

    for i in range(offset_h, img_h - offset_h):
        for j in range(offset_w, img_w - offset_w):
            region = image[i - offset_h:i + offset_h + 1, j - offset_w:j + offset_w + 1]
            if np.any(region == 255):
                output_image[i, j] = 255
            else:
                output_image[i, j] = 0
    return output_image

def opening(image, kernel):
    # Thực hiện erosion rồi dilation
    eroded_image = erosion(image, kernel)
    opened_image = dilation(eroded_image, kernel)
    return opened_image

# Tạo ảnh nhị phân mẫu
image = np.array([[0, 0, 0, 0, 0],
                  [0, 255, 255, 255, 0],
                  [0, 255, 255, 255, 0],
                  [0, 255, 255, 255, 0],
                  [0, 0, 0, 0, 0]], dtype=np.uint8)

# Tạo kernel
kernel = np.ones((3, 3), dtype=np.uint8) * 255

# Áp dụng các phép toán
eroded_image = erosion(image, kernel)
dilated_image = dilation(image, kernel)
opened_image = opening(image, kernel)

# Hiển thị kết quả
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray", vmin=0, vmax=255)

plt.subplot(2, 2, 2)
plt.title("Erosion")
plt.imshow(eroded_image, cmap="gray", vmin=0, vmax=255)

plt.subplot(2, 2, 3)
plt.title("Dilation")
plt.imshow(dilated_image, cmap="gray", vmin=0, vmax=255)

plt.subplot(2, 2, 4)
plt.title("Opening")
plt.imshow(opened_image, cmap="gray", vmin=0, vmax=255)

plt.show()
