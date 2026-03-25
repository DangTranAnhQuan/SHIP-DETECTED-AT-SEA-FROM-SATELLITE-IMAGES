import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import json # Thư viện mới để lưu số liệu
from shapely.geometry import Polygon
from mmdet.apis import init_detector, inference_detector
from mmrotate.utils import register_all_modules
import warnings

# Tắt các cảnh báo phiền phức
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

register_all_modules()

# ==============================================================================
# 🔴 CẤU HÌNH ĐƯỜNG DẪN
# ==============================================================================
CONFIG_FILE = r"D:\ShipDetectionWeb\models\r3det\config.py"
CHECKPOINT_FILE = r"D:\ShipDetectionWeb\models\r3det\weights.pth"

TEST_IMG_DIR = r"D:\ShipDetectionWeb\static\images\test\images"
TEST_LABEL_DIR = r"D:\ShipDetectionWeb\static\images\test\labelTxt"

# Tên file ảnh đầu ra (Lưu vào thư mục project)
OUTPUT_FILE = "r3det_pr_curve.png"
# File JSON lưu các số liệu quan trọng để cập nhật web
AP_METRICS_FILE = "r3det_ap_metrics.json"

# Ngưỡng IoU để coi là "Trúng"
IOU_THRESHOLD = 0.5


# ==============================================================================
# HÀM XỬ LÝ: ĐỌC LABEL DOTA (GROUND TRUTH)
# ==============================================================================
def parse_dota_label(txt_path):
    """Đọc file labelTxt trả về list các Polygon"""
    polygons = []
    if not os.path.exists(txt_path):
        return []

    with open(txt_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            # DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 classname difficulty
            if len(data) >= 8:
                try:
                    # Lấy 8 tọa độ đầu tiên tạo thành đa giác
                    coords = [float(x) for x in data[:8]]
                    poly = Polygon([(coords[0], coords[1]),
                                    (coords[2], coords[3]),
                                    (coords[4], coords[5]),
                                    (coords[6], coords[7])])
                    polygons.append(poly)
                except:
                    continue
    return polygons


# ==============================================================================
# HÀM XỬ LÝ: TÍNH IOU (INTERSECTION OVER UNION)
# ==============================================================================
def calculate_iou(poly1, poly2):
    """Tính IoU giữa 2 đa giác Shapely"""
    if not poly1.is_valid or not poly2.is_valid: return 0.0
    try:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        if union_area == 0: return 0.0
        return inter_area / union_area
    except:
        return 0.0


# ==============================================================================
# MAIN: CHẠY ĐÁNH GIÁ & VẼ BIỂU ĐỒ
# ==============================================================================
def main():
    print("⏳ Đang khởi tạo Model R3Det...")
    try:
        model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0')
    except:
        print("⚠️ Không tìm thấy GPU, chạy bằng CPU (sẽ chậm hơn)...")
        model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cpu')

    print(f"🔄 Đang quét dữ liệu test tại: {TEST_IMG_DIR}")
    img_paths = glob.glob(os.path.join(TEST_IMG_DIR, "*.*"))
    valid_imgs = [p for p in img_paths if p.lower().endswith(('.png', '.jpg'))]

    if not valid_imgs:
        print("❌ LỖI: Không tìm thấy ảnh nào trong folder test/images")
        return

    # KÍCH HOẠT CHẾ ĐỘ LẤY MẪU NHANH (NẾU CÓ)
    # Lấy mẫu ngẫu nhiên 150 ảnh nếu tổng số > 150 (như đã thống nhất)
    import random
    if len(valid_imgs) > 150:
        print(f"⚡ Kích hoạt chế độ lấy mẫu nhanh (150/{len(valid_imgs)} ảnh)...")
        random.seed(42)
        random.shuffle(valid_imgs)
        valid_imgs = valid_imgs[:150]
        print(f"   --> Chỉ chạy trên {len(valid_imgs)} ảnh để tiết kiệm thời gian.")


    detections = []
    total_ground_truths = 0

    print(f"🚀 Bắt đầu đánh giá trên {len(valid_imgs)} ảnh...")

    for i, img_path in enumerate(valid_imgs):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(TEST_LABEL_DIR, basename + ".txt")

        # 2. Lấy Ground Truth (GT)
        gt_polys = parse_dota_label(label_path)
        total_ground_truths += len(gt_polys)

        gt_matched = [False] * len(gt_polys)

        # 3. Chạy Model lấy Prediction
        try:
            result = inference_detector(model, img_path)

            if hasattr(result, 'pred_instances'):
                scores = result.pred_instances.scores.cpu().numpy()
                bboxes = result.pred_instances.bboxes.cpu().numpy()
            else:
                dets = result[0]
                scores = dets[:, 5]
                bboxes = dets[:, :5]

            # 4. So khớp Prediction với Ground Truth
            for j, score in enumerate(scores):
                xc, yc, w, h, a = bboxes[j]
                rect = ((xc, yc), (w, h), a * 180 / np.pi)
                box_points = cv2.boxPoints(rect)
                pred_poly = Polygon(box_points)

                best_iou = 0
                best_gt_idx = -1

                for k, gt_poly in enumerate(gt_polys):
                    iou = calculate_iou(pred_poly, gt_poly)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = k

                is_tp = False
                if best_iou >= IOU_THRESHOLD:
                    if not gt_matched[best_gt_idx]:
                        is_tp = True
                        gt_matched[best_gt_idx] = True
                    else:
                        is_tp = False # Duplicate

                detections.append({'score': score, 'is_tp': is_tp})

        except Exception as e:
            # print(f"⚠️ Lỗi xử lý ảnh {basename}: {e}") # Bỏ dòng này để console sạch
            continue

        if i % 50 == 0:
            print(f"   Processed {i}/{len(valid_imgs)} images...")

    # ==========================================================================
    # TÍNH TOÁN PRECISION - RECALL VÀ AP
    # ==========================================================================
    detections.sort(key=lambda x: x['score'], reverse=True)

    tps = [1 if d['is_tp'] else 0 for d in detections]
    fps = [1 if not d['is_tp'] else 0 for d in detections]

    tp_cumsum = np.cumsum(tps)
    fp_cumsum = np.cumsum(fps)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / (total_ground_truths + 1e-6)

    # Tính Average Precision (AP)
    ap = np.trapz(precisions, recalls)

    print(f"\n📊 KẾT QUẢ THỰC TẾ:")
    print(f"   - Tổng số tàu thật (Ground Truth): {total_ground_truths}")
    print(f"   - Average Precision (AP): {ap:.4f}")

    # --------------------------------------------------------------------------
    # 🔴 LƯU SỐ LIỆU AP VÀO JSON ĐỂ CẬP NHẬT WEB
    # --------------------------------------------------------------------------
    metrics_data = {
        "model": "R3Det",
        "mAP": round(ap * 100, 2),
        "AP_IOU_0_50": round(ap * 100, 2) # Dùng cho mAP50 trên web
    }
    with open(AP_METRICS_FILE, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"✅ Đã lưu số liệu AP vào: {os.path.abspath(AP_METRICS_FILE)}")

    # ==========================================================================
    # VẼ BIỂU ĐỒ
    # ==============================================================================
    BG_COLOR = '#0a192f'
    TEXT_COLOR = '#ccd6f6'
    LINE_COLOR = '#64ffda'

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.plot(recalls, precisions, color=LINE_COLOR, linewidth=3, label=f'R3Det (AP={ap:.2f})')
    ax.fill_between(recalls, precisions, color=LINE_COLOR, alpha=0.15)

    ax.set_title('PRECISION-RECALL CURVE: R3DET', fontsize=16, fontweight='bold', color=TEXT_COLOR, pad=20)
    ax.set_xlabel('Recall (Độ phủ)', fontsize=12, fontweight='bold', color=TEXT_COLOR)
    ax.set_ylabel('Precision (Độ chính xác)', fontsize=12, fontweight='bold', color=TEXT_COLOR)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    ax.grid(True, linestyle='--', color='#112240', alpha=0.5)
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.spines['left'].set_color(TEXT_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=TEXT_COLOR)

    leg = ax.legend(loc="lower left", facecolor=BG_COLOR, edgecolor=TEXT_COLOR)
    for text in leg.get_texts(): text.set_color(TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, facecolor=BG_COLOR)
    print(f"✅ Đã lưu ảnh: {os.path.abspath(OUTPUT_FILE)}")
    plt.show()


if __name__ == "__main__":
    main()