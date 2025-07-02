import numpy as np
import math
from PIL import Image
import scipy.fft
import cv2  # 用于 PSNR 计算

def embed_DCT_blue(pic, mark):
    block_width = 8

    # 转换为 NumPy 数组
    img_array = np.array(pic)
    mark_array = np.array(mark)

    row, col = mark_array.shape

    # 分离 R、G、B 三个通道
    r_channel = img_array[:, :, 0]
    g_channel = img_array[:, :, 1]
    b_channel = img_array[:, :, 2]

    for i in range(row):
        for j in range(col):
            # 提取当前块（仅用于 B 通道）
            BLOCK = np.float32(b_channel[i * block_width:(i + 1) * block_width,
                                        j * block_width:(j + 1) * block_width])

            # DCT 变换
            BLOCK = scipy.fft.dct(BLOCK)

            # 根据水印位调整 DCT 系数
            a = -1 if mark_array[i][j] else 1
            BLOCK = BLOCK * (1 + a * 0.03)

            # IDCT 还原并更新 B 通道
            BLOCK = scipy.fft.idct(BLOCK).astype(np.uint8)
            b_channel[i * block_width:(i + 1) * block_width,
                     j * block_width:(j + 1) * block_width] = BLOCK

    # 合并三个通道
    merged = np.stack([r_channel, g_channel, b_channel], axis=2)
    return Image.fromarray(merged)

def extract_DCT_blue(pic, marked):
    block_width = 8

    # 获取原始和带水印图像的 NumPy 数组
    original_array = np.array(pic)
    marked_array = np.array(marked)

    # 仅提取蓝色通道
    original_b = original_array[:, :, 2]
    marked_b = marked_array[:, :, 2]

    row = original_b.shape[0] // block_width
    col = original_b.shape[1] // block_width

    decode_pic = np.zeros((row, col), dtype=bool)

    for i in range(row):
        for j in range(col):
            BLOCK_ORIGIN = np.float32(original_b[i*block_width:(i+1)*block_width,
                                                 j*block_width:(j+1)*block_width])
            BLOCK_MARKED = np.float32(marked_b[i*block_width:(i+1)*block_width,
                                               j*block_width:(j+1)*block_width])

            BLOCK_ORIGIN = scipy.fft.idct(BLOCK_ORIGIN)
            BLOCK_MARKED = scipy.fft.idct(BLOCK_MARKED)

            bo = BLOCK_ORIGIN[1, 1]
            bm = BLOCK_MARKED[1, 1]
            a = bm / bo - 1
            decode_pic[i, j] = a < 0

    return Image.fromarray(decode_pic)

# 新增：归一化相关系数 NC
def nc(original, extracted):
    original = np.array(original).astype(np.float64)
    extracted = np.array(extracted).astype(np.float64)
    return np.corrcoef(original.flatten(), extracted.flatten())[0, 1]


# 新增：PSNR 计算
def psnr(original, watermarked):
    mse = np.mean((original.astype("float") - watermarked.astype("float")) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


if __name__ == '__main__':
    # 打开载体图像（lena.jpg）
    pic = Image.open('lena.jpg').convert('RGB')  # 确保为 RGB 图像

    # 打开水印图像（cuc.jpg），并调整尺寸
    mark = Image.open('cuc.jpg').convert('L')
    mark = mark.resize((80, 80))
    mark = mark.point(lambda x: 0 if x < 128 else 255, '1')  # 二值化处理

    # 嵌入水印（仅在蓝色通道）
    pic_marked = embed_DCT_blue(pic, mark)
    pic_marked.save('DCT_pic_marked.png')

    # 提取水印（注意：提取时也应针对蓝色通道操作）
    ext_mark = extract_DCT_blue(pic, pic_marked)
    ext_mark.save('DCT_ext_mark.png')

    # 归一化相关系数 NC
    nc_value = nc(mark, ext_mark)
    print(f"✅ NC (Normalized Correlation): {nc_value:.4f}")

    # PSNR
    original_array = np.array(pic)
    watermarked_array = np.array(pic_marked)
    psnr_value = psnr(original_array, watermarked_array)
    print(f"✅ PSNR: {psnr_value:.2f} dB")
