import cv2

def convert_to_binary(input_path, output_path, threshold=240):
    # 读取灰度图像
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # 应用全局阈值处理，转换为二值图像
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # 保存结果
    cv2.imwrite(output_path, binary_image)

    # 显示图像（可选）
    cv2.imshow("Binary Image", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 示例调用
input_image = 'cuc.jpg'
output_image = 'cuc_binary.jpg'

convert_to_binary(input_image, output_image)

print(f"已保存二值图像至 {output_image}")
