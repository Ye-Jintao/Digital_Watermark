import cv2
import numpy as np

def embed_watermark(cover_path, watermark_path, output_path):
    # 读取原始图像和水印图像
    cover = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    # 获取图像尺寸
    Mc, Nc = cover.shape
    Mm, Nm = watermark.shape

    # 将水印在高度方向循环复制，填满原始图像的高度
    expanded_watermark = np.zeros_like(cover)
    repeat_times = (Mc + Mm - 1) // Mm  # 计算需要复制多少次
    tiled_watermark = np.tile(watermark, (repeat_times, 1))  # 沿高度方向复制

    # 截取前 Mc 行，确保与 cover 图像高度一致
    expanded_watermark = tiled_watermark[:Mc, :]

    # LSB 嵌入：将 cover 图像的最低有效位替换为水印
    watermarked_image = cover.copy()
    for i in range(Mc):
        for j in range(Nc):
            # 修改最低有效位
            watermarked_image[i, j] = (cover[i, j] & ~1) | (expanded_watermark[i, j] // 255)

    # 保存嵌入水印后的图像
    cv2.imwrite(output_path, watermarked_image)

    # 计算 PSNR
    psnr = cv2.PSNR(cover, watermarked_image)
    return psnr, watermarked_image


# 示例调用
cover_image = 'lena.jpg'
watermark_image = 'cuc.jpg'
output_image = 'lsb_watermarked.bmp'

psnr_value, result_image = embed_watermark(cover_image, watermark_image, output_image)

print(f"PSNR: {psnr_value:.2f} dB")

# 显示结果
cv2.imshow('Original', cv2.imread(cover_image, cv2.IMREAD_GRAYSCALE))
cv2.imshow('Watermarked Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
