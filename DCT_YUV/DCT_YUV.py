import cv2 as cv
import numpy as np

def insert_watermark(pic_path, mark_path, output_path="after_mark_dct.jpg"):
    # Load the cover image and watermark
    pic = cv.imread(pic_path)
    mark = cv.imread(mark_path, 0)  # Grayscale watermark

    # Threshold and resize watermark to match embeddable capacity
    ret, mark = cv.threshold(mark, 128, 255, cv.THRESH_BINARY)
    block_8x8_row = pic.shape[0] // 8
    block_8x8_col = pic.shape[1] // 8
    total_blocks = block_8x8_row * block_8x8_col
    r = 3  # Number of points per block (fixed)
    max_pixels = total_blocks * r
    mark_size = int(np.sqrt(max_pixels))  # Square watermark
    mark = cv.resize(mark, (mark_size, mark_size))

    # Convert image to YUV and work with float32
    pic_YUV = cv.cvtColor(pic, cv.COLOR_BGR2YUV).astype('float32')
    water_mark = pic_YUV.copy()

    # Fixed mid-frequency positions for embedding
    positions = [(2, 3), (3, 2), (4, 3)]  # Example positions
    assert len(positions) == r, "Number of positions must match r"

    # Flatten watermark for sequential access
    mark_flat = mark.flatten()
    mark_idx = 0
    delta = 10.0  # Fixed embedding strength

    for i in range(block_8x8_row):
        for j in range(block_8x8_col):
            block_dct = cv.dct(pic_YUV[8*i:8*i+8, 8*j:8*j+8, 0])
            if block_dct.shape != (8, 8):
                continue

            for x, y in positions:
                if mark_idx >= len(mark_flat):
                    break
                sym_x, sym_y = 7 - x, 7 - y
                dot = block_dct[x, y]
                sym_dot = block_dct[sym_x, sym_y]

                if mark_flat[mark_idx] == 0:  # Black pixel
                    if dot <= sym_dot + delta:
                        block_dct[x, y] = sym_dot + delta + 1
                else:  # White pixel
                    if sym_dot <= dot + delta:
                        block_dct[sym_x, sym_y] = dot + delta + 1

                mark_idx += 1

            water_mark[8*i:8*i+8, 8*j:8*j+8, 0] = cv.idct(block_dct)

            if mark_idx >= len(mark_flat):
                break
        if mark_idx >= len(mark_flat):
            break

    # Convert back to uint8 and RGB
    water_mark_uint8 = water_mark.astype('uint8')
    water_mark_rgb = cv.cvtColor(water_mark_uint8, cv.COLOR_YUV2BGR)
    cv.imwrite(output_path, water_mark_rgb, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    return output_path

def get_mark(water_mark_path, mark_path):
    # Load watermarked image
    water_mark_rgb = cv.imread(water_mark_path)
    water_mark_YUV = cv.cvtColor(water_mark_rgb, cv.COLOR_BGR2YUV).astype('float32')

    # Calculate watermark size based on blocks
    block_8x8_row = water_mark_YUV.shape[0] // 8
    block_8x8_col = water_mark_YUV.shape[1] // 8
    total_blocks = block_8x8_row * block_8x8_col
    r = 3  # Must match embedding
    max_pixels = total_blocks * r
    mark_size = int(np.sqrt(max_pixels))
    finish_water_mark = np.zeros((mark_size, mark_size), np.uint8)

    # Fixed positions (same as embedding)
    positions = [(2, 3), (3, 2), (4, 3)]

    mark_idx = 0

    for i in range(block_8x8_row):
        for j in range(block_8x8_col):
            block_dct = cv.dct(water_mark_YUV[8*i:8*i+8, 8*j:8*j+8, 0])
            if block_dct.shape != (8, 8):
                continue

            for x, y in positions:
                if mark_idx >= mark_size * mark_size:
                    break
                sym_x, sym_y = 7 - x, 7 - y
                dot = block_dct[x, y]
                sym_dot = block_dct[sym_x, sym_y]

                # Extract based on coefficient relationship
                if dot > sym_dot:
                    finish_water_mark[mark_idx // mark_size, mark_idx % mark_size] = 0  # Black
                else:
                    finish_water_mark[mark_idx // mark_size, mark_idx % mark_size] = 255  # White

                mark_idx += 1

            if mark_idx >= mark_size * mark_size:
                break
        if mark_idx >= mark_size * mark_size:
            break

    cv.imwrite(mark_path, finish_water_mark, [int(cv.IMWRITE_JPEG_QUALITY), 100])

# NC and PSNR functions (unchanged)
def nc(original, extracted):
    original = cv.resize(original, (extracted.shape[1], extracted.shape[0]))
    original = original.astype(np.float64).flatten()
    extracted = extracted.astype(np.float64).flatten()
    return np.corrcoef(original, extracted)[0, 1]

def psnr(original, watermarked):
    mse = np.mean((original - watermarked) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else float('inf')

if __name__ == '__main__':
    cover_path = 'lena.jpg'
    watermark_path = 'cuc.jpg'
    watermarked_path = 'after_mark_dct.jpg'
    extracted_path = 'get_mark_dct.jpg'

    # Embed and extract watermark
    insert_watermark(cover_path, watermark_path, watermarked_path)
    get_mark(watermarked_path, extracted_path)

    # Evaluate
    original_mark = cv.imread(watermark_path, 0)
    extracted_mark = cv.imread(extracted_path, 0)
    original_cover = cv.imread(cover_path)
    watermarked_cover = cv.imread(watermarked_path)

    nc_value = nc(original_mark, extracted_mark)
    psnr_value = psnr(original_cover, watermarked_cover)

    print(f"✅ NC (Normalized Correlation): {nc_value:.4f}")
    print(f"✅ PSNR: {psnr_value:.2f} dB")