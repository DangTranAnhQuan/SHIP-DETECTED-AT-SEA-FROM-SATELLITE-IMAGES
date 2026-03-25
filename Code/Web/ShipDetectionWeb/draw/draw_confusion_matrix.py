import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import json  # Thư viện mới để lưu/đọc TP/FP/FN
from shapely.geometry import Polygon
from mmdet.apis import init_detector, inference_detector
from mmrotate.utils import register_all_modules
import warnings

register_all_modules()

# Tắt cảnh báo để màn hình hiển thị sạch sẽ
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==============================================================================
# 🔴 CẤU HÌNH ĐƯỜNG DẪN VÀ THAM SỐ
# ==============================================================================
CONFIG_FILE = r"D:\ShipDetectionWeb\models\r3det\config.py"
CHECKPOINT_FILE = r"D:\ShipDetectionWeb\models\r3det\weights.pth"
TEST_IMG_DIR = r"D:\ShipDetectionWeb\static\images\test\images"
TEST_LABEL_DIR = r"D:\ShipDetectionWeb\static\images\test\labelTxt"

OUTPUT_FILE = "../static/images/r3det_confusion_matrix.png"
METRICS_FILE = "../data/r3det_metrics.json"  # File lưu TP, FP, FN

IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.3

# Màu sắc OceanEye
BG_COLOR = '#0a192f'
LINE_COLOR = '#64ffda'
TEXT_COLOR = '#ccd6f6'


# ==============================================================================
# HÀM XỬ LÝ DỮ LIỆU
# ==============================================================================
def parse_dota_label(txt_path):
    """Đọc file labelTxt trả về list các Polygon"""
    polygons = []
    if not os.path.exists(txt_path): return []
    with open(txt_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) >= 8:
                try:
                    coords = [float(x) for x in data[:8]]
                    poly = Polygon([(coords[0], coords[1]), (coords[2], coords[3]),
                                    (coords[4], coords[5]), (coords[6], coords[7])])
                    polygons.append(poly)
                except:
                    continue
    return polygons


def calculate_iou(poly1, poly2):
    """Tính IoU giữa 2 đa giác Shapely"""
    if not poly1.is_valid or not poly2.is_valid: return 0.0
    try:
        inter = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - inter
        return inter / union if union > 0 else 0.0
    except:
        return 0.0


# ==============================================================================
# HÀM CHÍNH (TÍNH TOÁN)
# ==============================================================================
def run_full_calculation():
    """Chạy toàn bộ quá trình inference và tính TP/FP/FN."""
    print("⏳ Đang khởi tạo R3Det...")
    try:
        model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0')
    except:
        print("⚠️ Không tìm thấy GPU, chạy bằng CPU (sẽ chậm hơn)...")
        model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cpu')

    img_paths = glob.glob(os.path.join(TEST_IMG_DIR, "*.*"))
    valid_imgs = [p for p in img_paths if p.lower().endswith(('.png', '.jpg'))]

    print(f"🚀 Đang tính toán Confusion Matrix trên TẤT CẢ {len(valid_imgs)} ảnh...")

    TP = 0
    FP = 0
    FN = 0

    for i, img_path in enumerate(valid_imgs):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(TEST_LABEL_DIR, basename + ".txt")

        # 1. Lấy Ground Truth (Tàu thật)
        gt_polys = parse_dota_label(label_path)
        num_gt = len(gt_polys)
        gt_matched = [False] * num_gt

        # 2. Lấy Prediction (Tàu dự đoán)
        try:
            result = inference_detector(model, img_path)
            if hasattr(result, 'pred_instances'):
                scores = result.pred_instances.scores.cpu().numpy()
                bboxes = result.pred_instances.bboxes.cpu().numpy()
            else:
                dets = result[0]
                scores = dets[:, 5]
                bboxes = dets[:, :5]
        except:
            continue

        # Lọc box có điểm thấp
        keep_idxs = scores > SCORE_THRESHOLD
        scores = scores[keep_idxs]
        bboxes = bboxes[keep_idxs]

        # 3. So khớp
        for j, score in enumerate(scores):
            xc, yc, w, h, a = bboxes[j]
            rect = ((xc, yc), (w, h), a * 180 / np.pi)
            pred_poly = Polygon(cv2.boxPoints(rect))

            best_iou = 0
            best_gt_idx = -1

            for k, gt_poly in enumerate(gt_polys):
                iou = calculate_iou(pred_poly, gt_poly)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = k

            if best_iou >= IOU_THRESHOLD:
                if not gt_matched[best_gt_idx]:
                    TP += 1
                    gt_matched[best_gt_idx] = True
                else:
                    FP += 1  # Duplicate detection (Vẫn tính là False Positive)
            else:
                FP += 1  # Báo động giả (Nhận nhầm background thành tàu)

        # Số tàu thật không được match là False Negative (Bỏ sót)
        FN += gt_matched.count(False)

        if (i + 1) % 50 == 0 or i == len(valid_imgs) - 1:
            print(f"   Processed {i + 1}/{len(valid_imgs)} images...")

    # -------------------------------------------------------------
    # LƯU KẾT QUẢ VÀO FILE JSON
    # -------------------------------------------------------------
    metrics = {
        "model": "R3Det",
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "IOU_THRESHOLD": IOU_THRESHOLD,
        "SCORE_THRESHOLD": SCORE_THRESHOLD
    }

    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\n✅ Đã tính toán xong! Kết quả lưu tại: {os.path.abspath(METRICS_FILE)}")
    return METRICS_FILE


# ==============================================================================
# HÀM VẼ (TÁCH BIỆT)
# ==============================================================================
def draw_heatmap_from_file(metrics_file):
    """Chỉ vẽ biểu đồ từ file JSON đã lưu."""
    if not os.path.exists(metrics_file):
        print(f"❌ LỖI: Không tìm thấy file metrics {metrics_file}. Vui lòng chạy lần đầu để tính toán.")
        return

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    TP = metrics['TP']
    FP = metrics['FP']
    FN = metrics['FN']

    # ==========================================================================
    # VẼ CONFUSION MATRIX (CODE VẼ Y HỆT BẠN ĐANG CÓ)
    # ==========================================================================
    matrix_data = np.array([[TP, FN], [FP, 0]])

    plt.figure(figsize=(8, 6), facecolor=BG_COLOR)
    ax = plt.axes()
    ax.set_facecolor(BG_COLOR)

    # 1. Chuẩn hóa cho hàng Tàu (Tính Recall và Miss Rate)
    row1_sum = TP + FN
    tp_pct = TP / row1_sum if row1_sum > 0 else 0
    fn_pct = FN / row1_sum if row1_sum > 0 else 0

    # 2. Dữ liệu hiển thị trong ô
    annot_labels = np.array([
        [f"TP: {TP}\n({tp_pct:.1%})", f"FN: {FN}\n({fn_pct:.1%})"],
        [f"FP: {FP}\n(False Alarm)", ""]
    ])

    # 3. Vẽ Heatmap
    sns.heatmap(matrix_data, annot=annot_labels, fmt='', cmap='viridis',
                cbar=False, linewidths=1, linecolor=LINE_COLOR,
                xticklabels=['Predicted Ship (Tàu)', 'Missed / Background (Bỏ Sót)'],
                yticklabels=['Actual Ship (Thực tế là tàu)', 'Actual Background (Thực tế là nền)'])

    # 4. Trang trí
    for text in ax.texts: text.set_color(TEXT_COLOR)

    plt.title('CONFUSION MATRIX: R3DET', fontsize=16, fontweight='bold', color=TEXT_COLOR, pad=20)
    plt.ylabel('Ground Truth (Thực tế)', fontsize=12, color=LINE_COLOR)
    plt.xlabel('Prediction (Dự đoán)', fontsize=12, color=LINE_COLOR)

    ax.tick_params(colors=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, facecolor=BG_COLOR)
    print(f"\n✅ Đã vẽ xong Confusion Matrix! TP={TP}, FP={FP}, FN={FN}")
    plt.show()


# ==============================================================================
# CHẠY SCRIPT
# ==============================================================================
if __name__ == "__main__":
    # --- CHẾ ĐỘ CHẠY ---
    # Đổi sang True nếu muốn CHẠY LẠI TÍNH TOÁN (chờ lâu)
    # Đổi sang False nếu muốn CHỈ VẼ LẠI (chạy nhanh)
    FORCE_RECALCULATE = False

    if FORCE_RECALCULATE:
        # Lần chạy đầu tiên: Tính toán, lưu file JSON và vẽ.
        run_full_calculation()
    else:
        # Lần chạy sau: Chỉ đọc file JSON đã lưu và vẽ lại.
        print("⚠️ Đang chạy chế độ VẼ LẠI từ file metrics đã lưu. Đặt FORCE_RECALCULATE = True để tính lại toàn bộ.")
        draw_heatmap_from_file(METRICS_FILE)