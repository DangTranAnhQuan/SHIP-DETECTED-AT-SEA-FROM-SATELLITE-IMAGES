import cv2
import numpy as np
import os
import math

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ (CONFIG)
# ==========================================
CONF_GAMMA = 1.2
CONF_CLAHE_CLIP = 2.0
CONF_CLAHE_GRID = (8, 8)
CONF_GAUSSIAN_KERNEL = (5, 5)

CONF_INPUT_SIZE = (1024, 1024)
CONF_OVERLAP = 0.2
MIN_LAND_AREA = 15000


# ==========================================
# 2. CÁC HÀM XỬ LÝ CƠ BẢN
# ==========================================
def apply_gamma_correction(img, gamma=1.2):
    if gamma == 1.0: return img
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


# ==========================================
# 3. HÀM XỬ LÝ NÂNG CAO (ADVANCED)
# ==========================================
def apply_advanced_processing(img):
    """
    Trả về: (ảnh_đã_xử_lý, ảnh_minh_họa_chi_tiết)
    """
    # 1. Edge Preserving
    smooth = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)

    # 2. Top-Hat (Lấy chi tiết sáng nhỏ)
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)

    # Tạo ảnh minh họa (Visualization) - Tăng sáng để dễ nhìn
    vis = cv2.multiply(tophat, 3)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # 3. Cộng gộp & Sharpening
    tophat_bgr = cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR)
    enhanced = cv2.addWeighted(smooth, 1.0, tophat_bgr, 0.8, 0)

    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
    sharp = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

    return sharp, vis


# ==========================================
# 4. CÁC HÀM CẮT ẢNH & MASK
# ==========================================
def slice_image(img, target_size=(1024, 1024), overlap=0.2):
    h_img, w_img = img.shape[:2]
    h_target, w_target = target_size
    pad_h, pad_w = 0, 0
    if h_img < h_target or w_img < w_target:
        pad_h = max(0, h_target - h_img)
        pad_w = max(0, w_target - w_img)
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        h_img, w_img = img.shape[:2]

    stride_h = int(h_target * (1 - overlap))
    stride_w = int(w_target * (1 - overlap))
    patches = []
    y_steps = int(math.ceil((h_img - h_target) / stride_h)) + 1
    x_steps = int(math.ceil((w_img - w_target) / stride_w)) + 1

    for y in range(y_steps):
        for x in range(x_steps):
            y_start = y * stride_h
            x_start = x * stride_w
            if y_start + h_target > h_img: y_start = h_img - h_target
            if x_start + w_target > w_img: x_start = w_img - w_target
            patch = img[y_start:y_start + h_target, x_start:x_start + w_target].copy()
            patches.append({'img': patch, 'offset': (x_start, y_start)})
    return patches, img


def fill_smart_holes(land_mask):
    inv_land = cv2.bitwise_not(land_mask)
    contours, _ = cv2.findContours(inv_land, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = land_mask.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        touching_border = (x <= 2) or (y <= 2) or (x + bw >= w - 2) or (y + bh >= h - 2)
        if not touching_border and area < 3000:
            cv2.drawContours(land_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    return land_mask


def create_coastal_mask(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ksize = (7, 7)
    img32 = gray.astype(np.float32)
    mu = cv2.blur(img32, ksize)
    mu2 = cv2.blur(img32 * img32, ksize)
    sigma = np.sqrt(np.maximum(mu2 - mu * mu, 0))
    sigma = np.uint8(np.clip(sigma, 0, 255))
    _, mask_texture = cv2.threshold(sigma, 20, 255, cv2.THRESH_BINARY)
    blur_bright = cv2.GaussianBlur(gray, (25, 25), 0)
    _, mask_bright = cv2.threshold(blur_bright, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw_land_mask = cv2.bitwise_or(mask_texture, mask_bright)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed_mask = cv2.morphologyEx(raw_land_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    final_land_mask = np.zeros((h, w), dtype=np.uint8)
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        touching_border = (x <= 5) or (y <= 5) or (x + bw >= w - 5) or (y + bh >= h - 5)
        if area > MIN_LAND_AREA:
            cv2.drawContours(final_land_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        elif area > 5000 and touching_border:
            cv2.drawContours(final_land_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    final_land_mask = fill_smart_holes(final_land_mask)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_land_mask = cv2.dilate(final_land_mask, kernel_dilate, iterations=2)
    water_mask = cv2.bitwise_not(final_land_mask)
    return water_mask


# ==========================================
# 5. PIPELINE CHÍNH (CẬP NHẬT TRẢ VỀ 7 GIÁ TRỊ)
# ==========================================
def preprocess_image_for_inference(image_input, use_advanced_proc=False):
    try:
        img = None
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                # [FIXED] Trả về 7 giá trị khi lỗi
                return None, None, None, None, None, None, f"File không tồn tại"
            img = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            img = image_input

        if img is None:
            # [FIXED] Trả về 7 giá trị khi lỗi
            return None, None, None, None, None, None, "Lỗi đọc ảnh"

        original_h, original_w = img.shape[:2]
        original_shape = (original_h, original_w)

        detail_vis = None  # Biến chứa ảnh chi tiết (Visualization)

        if use_advanced_proc:
            # Nhận về 2 ảnh: Ảnh xử lý & Ảnh minh họa
            img_proc, detail_vis = apply_advanced_processing(img)
            img_proc = apply_clahe(img_proc, clip_limit=1.5)
        else:
            img_proc = cv2.GaussianBlur(img, CONF_GAUSSIAN_KERNEL, 0)
            img_proc = apply_clahe(img_proc, clip_limit=CONF_CLAHE_CLIP, tile_grid_size=CONF_CLAHE_GRID)
            img_proc = apply_gamma_correction(img_proc, gamma=CONF_GAMMA)

        coastal_mask = create_coastal_mask(img)
        img_to_slice = img_proc if use_advanced_proc else img
        patches_data, padded_img_original = slice_image(img_to_slice, CONF_INPUT_SIZE, CONF_OVERLAP)

        h_pad, w_pad = padded_img_original.shape[:2]

        def pad_to_match(src, th, tw):
            if src is None: return None
            h, w = src.shape[:2]
            if h < th or w < tw:
                return cv2.copyMakeBorder(src, 0, th - h, 0, tw - w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            return src

        original_display = pad_to_match(img, h_pad, w_pad)
        enhanced_display = pad_to_match(img_proc, h_pad, w_pad)
        coastal_mask_display = pad_to_match(coastal_mask, h_pad, w_pad)
        detail_vis_display = pad_to_match(detail_vis, h_pad, w_pad)  # Padding cho ảnh detail

        # [QUAN TRỌNG] TRẢ VỀ ĐÚNG 7 GIÁ TRỊ
        return patches_data, original_display, enhanced_display, coastal_mask_display, detail_vis_display, original_shape, None

    except Exception as e:
        # [FIXED] Trả về 7 giá trị khi Exception
        return None, None, None, None, None, None, f"Lỗi Preprocessing: {str(e)}"