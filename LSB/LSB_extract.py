import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def extract_watermark(watermarked_path, original_watermark_path, output_watermark_path='extracted_watermark.png'):
    # 读取原始水印图像（用于比较误比特率和相似度）
    orig_watermark = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
    if orig_watermark is None:
        raise FileNotFoundError(f"无法读取原始水印图像，请检查路径是否正确：{original_watermark_path}")

    print("✅ 原始水印图像已加载")

    # 获取原始水印尺寸
    Mm, Nm = orig_watermark.shape

    # 读取带水印的图像
    watermarked_image = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE)
    if watermarked_image is None:
        raise FileNotFoundError(f"无法读取带水印图像，请检查路径是否正确：{watermarked_image_path}")

    print("✅ 带水印图像已加载")
    Mw, Nw = watermarked_image.shape

    # 提取水印：从 LSB 中获取每一位
    extracted_watermark = np.zeros_like(watermarked_image)
    for i in range(Mw):
        for j in range(Nw):
            extracted_watermark[i, j] = np.uint8(watermarked_image[i, j] & 1)

    # 将提取的水印扩展为与原水印相同大小
    watermark1 = np.zeros((Mm, Nm), dtype=np.uint8)
    for i in range(Mm - 1):
        for j in range(Nm - 1):
            watermark1[i + 1, j + 1] = extracted_watermark[i % Mw, j % Nw]
    watermark1[0, 0] = extracted_watermark[-1, -1]

    print("✅ 水印提取完成")

    # 保存提取出的水印图像
    cv2.imwrite(output_watermark_path, watermark1 * 255)  # 放大到 [0,255] 范围保存
    print(f"✅ 已保存提取出的水印图像至 {output_watermark_path}")

    # 显示提取出的水印图像
    cv2.imshow("Recovered Watermark", watermark1 * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 计算误比特率 BER
    message_pad = np.round(orig_watermark / 256.0).astype(np.uint8).flatten()
    recovered_flat = np.round(watermark1 / 256.0).astype(np.uint8).flatten()
    len_bits = min(len(message_pad), len(recovered_flat))  # 防止长度不一致
    bit_error_rate = np.sum(np.abs(recovered_flat[:len_bits] - message_pad[:len_bits])) / len_bits
    print(f"✅ 误比特率 (BER): {bit_error_rate:.4f}")

    # 缩放提取的水印以进行 SSIM 比较
    resized_watermark1 = cv2.resize(watermark1 * 255, (Nm, Mm), interpolation=cv2.INTER_AREA)

    # 确保输入图像有效
    assert resized_watermark1.shape == orig_watermark.shape, "图像尺寸不匹配，无法计算 SSIM"

    similarity = ssim(orig_watermark, resized_watermark1, data_range=255, multichannel=False, channel_axis=None)
    print(f"✅ 结构相似度 SSIM: {similarity:.4f}")

    return bit_error_rate, watermark1, similarity


# 示例调用
if __name__ == '__main__':
    watermarked_image_path = 'lsb_watermarked.bmp'
    original_watermark_path = 'cuc.jpg'

    try:
        print("🚀 开始提取水印...")
        ber, recovered_watermark, similarity = extract_watermark(watermarked_image_path, original_watermark_path)
    except Exception as e:
        print(f"❌ 发生错误：{e}")
