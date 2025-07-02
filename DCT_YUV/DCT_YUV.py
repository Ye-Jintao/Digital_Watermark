import numpy as np
import math
import os
from PIL import Image
import scipy.fft
import cv2
import glob


def embed_DCT_blue(pic, mark):
    """
    在图片的蓝色通道DCT系数中嵌入水印
    :param pic: 原始载体图像(PIL Image)
    :param mark: 二值水印图像(PIL Image, 模式'1')
    :return: 含水印的图像(PIL Image)
    """
    block_width = 8
    img_array = np.array(pic)
    mark_array = np.array(mark).astype(bool)
    row, col = mark_array.shape

    r_channel = img_array[:, :, 0]
    g_channel = img_array[:, :, 1]
    b_channel = img_array[:, :, 2].astype(np.float32)  # 使用float32进行计算

    for i in range(row):
        for j in range(col):
            # 获取当前8x8块
            BLOCK = b_channel[i * block_width:(i + 1) * block_width,
                    j * block_width:(j + 1) * block_width]
            # DCT变换
            BLOCK = scipy.fft.dct(BLOCK)
            # 修改[1,1]位置的系数
            if mark_array[i, j]:  # 水印位=1
                BLOCK[1, 1] += 30.0  # 增加系数
            else:  # 水印位=0
                BLOCK[1, 1] -= 30.0  # 减少系数
            # 逆DCT变换
            BLOCK = scipy.fft.idct(BLOCK)
            # 将处理后的块放回原位置
            b_channel[i * block_width:(i + 1) * block_width,
            j * block_width:(j + 1) * block_width] = BLOCK

    # 将蓝色通道数据转换回uint8并确保在0-255范围内
    b_channel = np.clip(b_channel, 0, 255).astype(np.uint8)

    # 合并通道
    merged = np.stack([r_channel, g_channel, b_channel], axis=2).astype(np.uint8)
    return Image.fromarray(merged)


def extract_DCT_blue(pic, marked):
    """
    从含水印图像的蓝色通道DCT系数中提取水印
    :param pic: 原始载体图像(PIL Image)
    :param marked: 含水印的图像(PIL Image)
    :return: 提取的水印图像(PIL Image)
    """
    block_width = 8
    original_array = np.array(pic)
    marked_array = np.array(marked)
    original_b = original_array[:, :, 2].astype(np.float32)
    marked_b = marked_array[:, :, 2].astype(np.float32)
    row = original_b.shape[0] // block_width
    col = original_b.shape[1] // block_width
    decode_pic = np.zeros((row, col), dtype=bool)

    for i in range(row):
        for j in range(col):
            # 获取原始图像和带水印图像的块
            BLOCK_ORIGIN = original_b[i * block_width:(i + 1) * block_width,
                           j * block_width:(j + 1) * block_width]
            BLOCK_MARKED = marked_b[i * block_width:(i + 1) * block_width,
                           j * block_width:(j + 1) * block_width]
            # 对两个块进行DCT变换
            BLOCK_ORIGIN = scipy.fft.dct(BLOCK_ORIGIN)
            BLOCK_MARKED = scipy.fft.dct(BLOCK_MARKED)
            # 比较[1,1]位置的系数
            bo = BLOCK_ORIGIN[1, 1]
            bm = BLOCK_MARKED[1, 1]
            # 如果带水印图像的系数更大，则判定为1
            decode_pic[i, j] = bm > bo

    return Image.fromarray(decode_pic)


def nc(original, extracted):
    """计算归一化相关系数(NC)"""
    original = np.array(original).astype(np.float64)
    extracted = np.array(extracted).astype(np.float64)
    return np.corrcoef(original.flatten(), extracted.flatten())[0, 1]


def psnr(original, watermarked):
    """计算PSNR(峰值信噪比)"""
    mse = np.mean((original.astype("float") - watermarked.astype("float")) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def resize_to_block_size(img, block_size=8):
    """调整图像尺寸为block_size的整数倍"""
    w, h = img.size
    new_w = (w // block_size) * block_size
    new_h = (h // block_size) * block_size
    return img.resize((new_w, new_h))


def find_image_file(base_name, extensions=None):
    """查找图像文件，支持多种扩展名和大小写"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']

    # 首先检查无扩展名的情况
    if os.path.exists(base_name):
        return base_name

    # 检查带扩展名的情况
    for ext in extensions:
        file_path = f"{base_name}{ext}"
        if os.path.exists(file_path):
            return file_path

    # 使用glob匹配更广泛的模式
    patterns = [
        f"{base_name}.*",
        f"{base_name.upper()}.*",
        f"{base_name.lower()}.*"
    ]

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            # 过滤出支持的图像扩展名
            for match in matches:
                _, ext = os.path.splitext(match)
                if ext.lower() in [e.lower() for e in extensions]:
                    return match

    return None


if __name__ == '__main__':
    # 打印当前目录以便调试
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    print("目录中的文件:")
    for f in os.listdir(current_dir):
        print(f" - {f}")

    # 查找载体和水印图像
    cover_path = find_image_file('lena')
    watermark_path = find_image_file('cuc')

    if not cover_path:
        raise FileNotFoundError("未找到载体图像，请确保有名为'lena'的图像文件")
    if not watermark_path:
        raise FileNotFoundError("未找到水印图像，请确保有名为'cuc'的图像文件")

    print(f"✅ 找到载体图像: {cover_path}")
    print(f"✅ 找到水印图像: {watermark_path}")

    _, cover_ext = os.path.splitext(cover_path)
    _, watermark_ext = os.path.splitext(watermark_path)

    # 预处理载体图像
    pic = Image.open(cover_path).convert('RGB')
    pic_resized = resize_to_block_size(pic)
    print(f"✅ 载体图像尺寸调整为: {pic_resized.size}")

    # 计算块尺寸
    block_width = 8
    target_blocks_h = pic_resized.height // block_width
    target_blocks_w = pic_resized.width // block_width

    # 预处理水印图像
    mark = Image.open(watermark_path).convert('L')
    mark = mark.point(lambda x: 0 if x < 128 else 1, mode='1')
    mark_resized = mark.resize((target_blocks_w, target_blocks_h))
    print(f"✅ 水印图像尺寸调整为: {mark_resized.size} (匹配 {target_blocks_h}x{target_blocks_w} 个块)")

    # 嵌入水印
    pic_marked = embed_DCT_blue(pic_resized, mark_resized)
    output_path = f'DCT_pic_marked{cover_ext}'
    pic_marked.save(output_path)
    print(f"✅ 含水印图像已保存为: {output_path}")

    # 提取水印
    ext_mark = extract_DCT_blue(pic_resized, pic_marked)
    print(f"✅ 提取的水印尺寸: {ext_mark.size}")

    # 调整提取的水印到原始尺寸并保存
    original_mark = Image.open(watermark_path).convert('1')
    final_ext_mark = ext_mark.resize(original_mark.size)
    final_output_path = f'DCT_ext_mark{watermark_ext}'
    final_ext_mark.save(final_output_path)
    print(f"✅ 提取的水印已调整为原始尺寸并保存为: {final_output_path}")

    # 计算NC和PSNR
    nc_value = nc(mark_resized, ext_mark)
    psnr_value = psnr(np.array(pic_resized), np.array(pic_marked))
    print(f"✅ NC值: {nc_value:.4f}, PSNR值: {psnr_value:.2f} dB")