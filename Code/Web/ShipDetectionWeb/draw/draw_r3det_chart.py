import json
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==============================================================================
# Đường dẫn đến file log của bạn
LOG_FILE_PATH = r"D:\results\work_dirs\r3det_ship_final_v18\20251213_174241\vis_data\20251213_174241.json"
OUTPUT_FILE = "../static/images/r3det_training_history.png"
MODEL_NAME = "R3Det (Refined Rotated RetinaNet)"

# --- MÀU SẮC THEME OCEANEYE ---
BG_COLOR = '#0a192f'  # Nền xanh đen đậm
TEXT_COLOR = '#ccd6f6'  # Chữ màu trắng xám
ACC_COLOR = '#64ffda'  # Màu Cyan (cho mAP)
LOSS_COLOR = '#ff6b6b'  # Màu Đỏ cam (cho Loss)
GRID_COLOR = '#112240'  # Màu lưới mờ


# ==============================================================================
# 2. HÀM XỬ LÝ FILE LOG (ĐÃ SỬA LỖI dota/mAP)
# ==============================================================================
def parse_mmrotate_log(log_path):
    print(f"🔄 Đang đọc file log: {log_path}")

    epoch_losses = {}
    epoch_maps = {}

    if not os.path.exists(log_path):
        print(f"❌ LỖI: Không tìm thấy file tại {log_path}")
        return None, None, None

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            try:
                log = json.loads(line)
            except json.JSONDecodeError:
                continue

            # --- SỬA LỖI 1: Lấy Epoch hoặc Step ---
            # Dòng train có 'epoch', dòng val có 'step'
            epoch = log.get('epoch', log.get('step'))
            if epoch is None: continue  # Bỏ qua nếu không xác định được thời điểm

            # --- SỬA LỖI 2: Bắt key 'dota/mAP' ---

            # 1. LẤY LOSS (Training)
            if 'loss' in log:
                if epoch not in epoch_losses:
                    epoch_losses[epoch] = []
                epoch_losses[epoch].append(log['loss'])

            # 2. LẤY mAP (Validation)
            mAP_value = None

            # Ưu tiên tìm key 'dota/mAP' như trong log của bạn
            if 'dota/mAP' in log:
                mAP_value = log['dota/mAP']
            elif 'coco/bbox_mAP' in log:
                mAP_value = log['coco/bbox_mAP']
            elif 'mAP' in log:
                mAP_value = log['mAP']

            if mAP_value is not None:
                # Nếu mAP nhỏ (vd: 0.79), nhân 100 để thành 79%
                # Nếu mAP lớn (vd: 79.0), giữ nguyên
                if mAP_value <= 1.0:
                    mAP_value *= 100
                epoch_maps[epoch] = mAP_value

    # --- TỔNG HỢP DỮ LIỆU ---
    all_epochs = sorted(list(set(epoch_losses.keys()) | set(epoch_maps.keys())))

    final_epochs = []
    final_losses = []
    final_maps = []

    # Biến tạm để vẽ đường mAP mượt hơn (giữ giá trị cũ nếu epoch đó không val)
    last_map = 0

    for ep in all_epochs:
        # Tính trung bình Loss
        avg_loss = None
        if ep in epoch_losses and len(epoch_losses[ep]) > 0:
            avg_loss = sum(epoch_losses[ep]) / len(epoch_losses[ep])

        # Lấy mAP
        current_map = epoch_maps.get(ep, None)

        # Logic điền dữ liệu:
        # Nếu epoch này có Loss, ta lấy Loss.
        # mAP nếu có thì lấy, không thì lấy của epoch trước (để vẽ đường thẳng ngang)
        if avg_loss is not None:
            final_epochs.append(ep)
            final_losses.append(avg_loss)

            if current_map is not None:
                final_maps.append(current_map)
                last_map = current_map
            else:
                final_maps.append(last_map)  # Giữ nguyên mAP cũ

    print(f"✅ Đã trích xuất: {len(final_epochs)} Epochs.")
    print(f"   - Loss cuối: {final_losses[-1]:.4f}")
    print(f"   - mAP cuối: {final_maps[-1]:.2f}%")

    return final_epochs, final_losses, final_maps


# ==============================================================================
# 3. HÀM VẼ BIỂU ĐỒ
# ==============================================================================
def draw_chart():
    epochs, losses, maps = parse_mmrotate_log(LOG_FILE_PATH)

    if not epochs or len(epochs) == 0:
        print("⚠️ Không có dữ liệu để vẽ.")
        return

    # Thiết lập khung biểu đồ
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax1.set_facecolor(BG_COLOR)

    # --- VẼ LOSS (TRỤC TRÁI) ---
    ax1.set_xlabel('Epochs (Số vòng lặp)', fontsize=12, color=TEXT_COLOR, fontweight='bold')
    ax1.set_ylabel('Training Loss (Sai số vị trí)', fontsize=12, color=LOSS_COLOR, fontweight='bold')
    line1 = ax1.plot(epochs, losses, color=LOSS_COLOR, marker='o', linewidth=2, label='Training Loss', markersize=5)
    ax1.tick_params(axis='y', labelcolor=LOSS_COLOR, colors=LOSS_COLOR)
    ax1.tick_params(axis='x', colors=TEXT_COLOR)
    ax1.grid(True, linestyle='--', which='major', color=GRID_COLOR, alpha=0.5)

    # --- VẼ mAP (TRỤC PHẢI) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('mAP Accuracy (%)', fontsize=12, color=ACC_COLOR, fontweight='bold')

    # Vẽ mAP (chỉ đánh dấu điểm vuông tại những nơi có thực sự Val)
    line2 = ax2.plot(epochs, maps, color=ACC_COLOR, marker='s', linewidth=2, label='mAP Score', markersize=5)

    ax2.tick_params(axis='y', labelcolor=ACC_COLOR, colors=ACC_COLOR)

    # Chỉnh viền biểu đồ
    for ax in [ax1, ax2]:
        ax.spines['bottom'].set_color(TEXT_COLOR)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color(LOSS_COLOR)
        ax.spines['right'].set_color(ACC_COLOR)

    # --- CHÚ THÍCH & TIÊU ĐỀ ---
    plt.title(f'HIỆU XUẤT HUẤN LUYỆN BẰNG R3DET', fontsize=16, color=TEXT_COLOR, fontweight='bold', pad=20)

    # Legend chung
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    leg = ax1.legend(lines, labels, loc='center right', facecolor=BG_COLOR, edgecolor=TEXT_COLOR)
    for text in leg.get_texts():
        text.set_color(TEXT_COLOR)

    # Annotate Best mAP (Lấy giá trị lớn nhất)
    if len(maps) > 0:
        max_map = max(maps)
        # Tìm epoch tương ứng với max map
        max_idx = maps.index(max_map)
        max_epoch = epochs[max_idx]

        ax2.annotate(f'Best: {max_map:.2f}%',
                     xy=(max_epoch, max_map),
                     xytext=(max_epoch, max_map - 5 if max_map > 80 else max_map + 5),
                     arrowprops=dict(facecolor=ACC_COLOR, shrink=0.05),
                     color=ACC_COLOR, fontweight='bold', ha='center')

    plt.tight_layout()

    # Lưu file
    plt.savefig(OUTPUT_FILE, dpi=300, facecolor=BG_COLOR)
    print(f"\n🎉 XONG! Ảnh biểu đồ đã lưu tại: {os.path.abspath(OUTPUT_FILE)}")
    plt.show()


if __name__ == "__main__":
    draw_chart()