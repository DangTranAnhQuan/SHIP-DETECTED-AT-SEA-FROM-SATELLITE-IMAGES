import os
import cv2
import torch
import numpy as np
import gc
import json
import time
import preprocessing

# ==============================================================================
# 🔴 [FIX 1] SỬA LỖI PYTORCH 2.6+ (WeightsUnpickler error & Future Warnings)
# ==============================================================================
_original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = safe_torch_load
print("✅ [System] Đã áp dụng bản vá lỗi PyTorch 2.6 (Force weights_only=False)")

# ==========================================
# 1. SETUP THƯ VIỆN & DEVICE
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_INFO_PATH = os.path.join(BASE_DIR, 'data', 'models_info.json')
LOADED_MODELS = {}

# [FIX 2] SETUP DEVICE CHUẨN
if torch.cuda.is_available():
    DEVICE_NAME = 'cuda:0'
    DEVICE = torch.device(DEVICE_NAME)
    print(f"🚀 HỆ THỐNG ĐANG SỬ DỤNG GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE_NAME = 'cpu'
    DEVICE = torch.device('cpu')
    print("⚠️ HỆ THỐNG ĐANG CHẠY TRÊN CPU (Sẽ chậm!)")

# --- MMRotate ---
try:
    import mmrotate
    try:
        from mmdet.apis import init_detector, inference_detector
    except ImportError:
        from mmrotate.apis import init_detector, inference_detector
    print("✅ [System] Đã nạp thư viện MMRotate/MMDet")
except ImportError:
    print("⚠️ [System] Không tìm thấy MMRotate.")
    def init_detector(*args, **kwargs): return None
    def inference_detector(*args, **kwargs): return []

# --- Ultralytics (YOLO) ---
try:
    from ultralytics import YOLO
    print("✅ [System] Đã nạp thư viện Ultralytics (YOLO)")
except ImportError:
    print("⚠️ [System] Không tìm thấy Ultralytics")
    YOLO = None

# --- Torchvision ---
try:
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    print("✅ [System] Đã nạp thư viện Torchvision")
except ImportError:
    print("⚠️ [System] Không tìm thấy Torchvision")


# ==========================================
# 2. HELPER FUNCTIONS: MODEL FACTORY
# ==========================================
def get_faster_rcnn_model(num_classes=2):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_mask_rcnn_model(num_classes):
    """
    Khởi tạo Mask R-CNN với số class tùy chỉnh.
    """
    if num_classes == 91:
        return maskrcnn_resnet50_fpn(weights=None)

    model = maskrcnn_resnet50_fpn(weights=None)
    # 1. Thay thế Box Predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # 2. Thay thế Mask Predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def load_models_config():
    if not os.path.exists(MODELS_INFO_PATH): return {}
    with open(MODELS_INFO_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_model(model_id):
    global LOADED_MODELS
    if model_id in LOADED_MODELS:
        return LOADED_MODELS[model_id]

    info_list = load_models_config()
    if model_id not in info_list:
        print(f"⚠️ Model ID '{model_id}' không tồn tại.")
        return None

    model_info = info_list[model_id]
    model_type = model_info.get('type', 'mmrotate')

    ckpt_path = os.path.join(BASE_DIR, model_info.get('weights_file', '').replace('/', os.sep))
    cfg_path = os.path.join(BASE_DIR, model_info.get('config_file', '').replace('/', os.sep))

    print(f"⏳ Đang khởi tạo model: {model_info['name']} trên {DEVICE_NAME.upper()}...")

    try:
        # --- YOLO ---
        if model_type == 'yolo':
            if YOLO is None: return None
            if not os.path.exists(ckpt_path): return None
            model = YOLO(ckpt_path)
            model.to(DEVICE_NAME)
            LOADED_MODELS[model_id] = {'model': model, 'type': 'yolo'}
            return LOADED_MODELS[model_id]

        # --- FASTER R-CNN ---
        elif model_type == 'torchvision':
            if not os.path.exists(ckpt_path): return None
            model = get_faster_rcnn_model(num_classes=2)
            state_dict = torch.load(ckpt_path, map_location=DEVICE)
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            LOADED_MODELS[model_id] = {'model': model, 'type': 'torchvision', 'device': DEVICE}
            return LOADED_MODELS[model_id]

        # --- MASK R-CNN ---
        elif model_type == 'mask_rcnn_torch':
            if not os.path.exists(ckpt_path): return None
            checkpoint = torch.load(ckpt_path, map_location=DEVICE)
            state_dict = checkpoint['model'] if (isinstance(checkpoint, dict) and 'model' in checkpoint) else checkpoint
            
            model = None
            try:
                # CÁCH 1: Thử load 2 classes
                model_2 = get_mask_rcnn_model(num_classes=2)
                model_2.load_state_dict(state_dict, strict=True)
                model = model_2
            except RuntimeError:
                print("⚠️ Mismatch detected. Fallback: Load model 91 classes (COCO structure)...")
                # CÁCH 2: Fallback 91 classes
                model_91 = get_mask_rcnn_model(num_classes=91)
                model_91.load_state_dict(state_dict, strict=False)
                model = model_91

            if model:
                model.to(DEVICE)
                model.eval()
                LOADED_MODELS[model_id] = {'model': model, 'type': 'mask_rcnn_torch', 'device': DEVICE}
                return LOADED_MODELS[model_id]
            return None

        # --- MMROTATE ---
        else:
            if os.path.exists(cfg_path) and os.path.exists(ckpt_path):
                model = init_detector(cfg_path, ckpt_path, device=DEVICE_NAME)
                LOADED_MODELS[model_id] = {'model': model, 'type': 'mmrotate'}
                return LOADED_MODELS[model_id]
            else:
                return None
    except Exception as e:
        print(f"❌ Lỗi load model {model_id}: {e}")
        return None


# ==========================================
# 3. HELPER FUNCTIONS: XỬ LÝ KẾT QUẢ
# ==========================================
def simple_nms_merge(all_results, iou_thr=0.3):
    if not all_results: return []
    all_results.sort(key=lambda x: x['score'], reverse=True)
    keep = []
    while all_results:
        best = all_results.pop(0)
        keep.append(best)
        remaining = []
        for other in all_results:
            rect1 = cv2.boundingRect(best['box']) if best['type'] == 'poly' else best['box']
            rect2 = cv2.boundingRect(other['box']) if other['type'] == 'poly' else other['box']
            xA = max(rect1[0], rect2[0])
            yA = max(rect1[1], rect2[1])
            xB = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
            yB = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = rect1[2] * rect1[3]
            boxBArea = rect2[2] * rect2[3]
            iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
            if iou < iou_thr: remaining.append(other)
        all_results = remaining
    return keep

def process_mmrotate_result(result, offset_x, offset_y, score_thr):
    detected = []
    if hasattr(result, 'pred_instances'):
        instances = result.pred_instances.cpu().numpy()
        for i, score in enumerate(instances.scores):
            if score < score_thr: continue
            xc, yc, w, h, angle = instances.bboxes[i]
            rect = ((xc + offset_x, yc + offset_y), (w, h), angle * 180 / np.pi)
            
            # [FIX 3] LỖI NUMPY 2.0 (AttributeError: module 'numpy' has no attribute 'int0')
            box_points = np.int32(cv2.boxPoints(rect))
            
            detected.append({'box': box_points, 'score': float(score), 'type': 'poly', 'label': f"{score:.2f}"})
    return detected

def process_yolo_result(results, offset_x, offset_y, score_thr):
    detected = []
    result = results[0]
    if result.boxes is None: return []
    for box in result.boxes:
        score = float(box.conf[0])
        if score < score_thr: continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        detected.append({
            'box': [x1 + offset_x, y1 + offset_y, x2-x1, y2-y1],
            'score': score,
            'type': 'rect',
            'label': f"YOLO: {score:.2f}"
        })
    return detected

def process_torchvision_result(predictions, offset_x, offset_y, score_thr):
    detected = []
    pred = predictions[0]
    boxes = pred['boxes'].cpu().detach().numpy()
    scores = pred['scores'].cpu().detach().numpy()
    labels = pred['labels'].cpu().detach().numpy()
    masks = None
    if 'masks' in pred:
        masks = pred['masks'].cpu().detach().numpy()

    for i, score in enumerate(scores):
        if score < score_thr: continue
        # Nếu model có 91 lớp, Tàu thường là class 1 (nếu train custom) hoặc class 9 (COCO). 
        # Giả định train custom dataset label=1.
        if labels[i] != 1: continue

        xmin, ymin, xmax, ymax = boxes[i].astype(int)
        obj = {
            'box': [xmin + offset_x, ymin + offset_y, (xmax - xmin), (ymax - ymin)],
            'score': float(score),
            'type': 'rect',
            'label': f"{score:.2f}"
        }

        if masks is not None:
            raw_mask = masks[i, 0]
            binary_mask = (raw_mask > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            shifted_contours = []
            for cnt in contours:
                cnt += [offset_x, offset_y]
                shifted_contours.append(cnt)
            obj['contours'] = shifted_contours
            obj['has_mask'] = True

        detected.append(obj)
    return detected

def is_valid_location(box_center, mask):
    if mask is None: return True
    x, y = int(box_center[0]), int(box_center[1])
    h, w = mask.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h: return False
    return mask[y, x] > 0


# ==========================================
# 4. MAIN PREDICT FUNCTION
# ==========================================
def predict_image(image_path, model_id, conf_threshold=0.3, use_land_mask=False, use_advanced_proc=False):
    patches_data, display_img, enhanced_img, coastal_mask, detail_vis, original_shape, error = \
        preprocessing.preprocess_image_for_inference(image_path, use_advanced_proc=use_advanced_proc)
    if error: return None, f"Lỗi: {error}"

    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    save_dir = os.path.join(BASE_DIR, 'static', 'uploads')
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    orig_h, orig_w = original_shape
    enhanced_filename = f"{name}_enhanced{ext}"
    if enhanced_img is not None:
        cv2.imencode(ext, enhanced_img[0:orig_h, 0:orig_w])[1].tofile(os.path.join(save_dir, enhanced_filename))

    mask_filename = None
    active_mask = None
    if coastal_mask is not None:
        mask_filename = f"{name}_mask{ext}"
        cv2.imencode(ext, coastal_mask[0:orig_h, 0:orig_w])[1].tofile(os.path.join(save_dir, mask_filename))
        if use_land_mask: active_mask = coastal_mask

    detail_filename = None
    if detail_vis is not None and use_advanced_proc:
        detail_filename = f"{name}_detail{ext}"
        cv2.imencode(ext, detail_vis[0:orig_h, 0:orig_w])[1].tofile(os.path.join(save_dir, detail_filename))

    model_data = get_model(model_id)
    if not model_data: return None, "Không load được model"

    model = model_data['model']
    m_type = model_data['type']
    all_detections = []

    try:
        print(f"🔄 Đang xử lý {len(patches_data)} mảnh cắt trên {DEVICE_NAME}...")
        for patch_info in patches_data:
            img_input = patch_info['img']
            off_x, off_y = patch_info['offset']

            if m_type == 'mmrotate':
                res = inference_detector(model, img_input)
                objs = process_mmrotate_result(res, off_x, off_y, score_thr=conf_threshold)
                all_detections.extend(objs)
            elif m_type == 'yolo':
                img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
                res = model(img_rgb, imgsz=1024, verbose=False, device=DEVICE_NAME)
                objs = process_yolo_result(res, off_x, off_y, score_thr=conf_threshold)
                all_detections.extend(objs)
            elif m_type == 'torchvision' or m_type == 'mask_rcnn_torch':
                img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
                img_tensor = torchvision.transforms.functional.to_tensor(img_rgb).unsqueeze(0).to(model_data['device'])
                with torch.no_grad():
                    pred = model(img_tensor)
                objs = process_torchvision_result(pred, off_x, off_y, score_thr=conf_threshold)
                all_detections.extend(objs)

        final_objects = simple_nms_merge(all_detections, iou_thr=0.2)
        final_result_img = display_img.copy()
        overlay_mask = final_result_img.copy()
        count = 0

        for obj in final_objects:
            if obj['type'] == 'poly':
                M = cv2.moments(obj['box'])
                cx, cy = (int(M['m10']/M['m00']), int(M['m01']/M['m00'])) if M['m00']!=0 else (obj['box'][0][0], obj['box'][0][1])
            else:
                x, y, w, h = obj['box']
                cx, cy = x + w//2, y + h//2

            if is_valid_location((cx, cy), active_mask):
                count += 1
                if obj.get('has_mask', False):
                    cv2.drawContours(overlay_mask, obj['contours'], -1, (255, 0, 255), -1)

                if obj['type'] == 'poly':
                    cv2.polylines(final_result_img, [obj['box']], True, (0, 165, 255), 2)
                    lbl_pos = tuple(obj['box'][0])
                else:
                    x, y, w, h = obj['box']
                    cv2.rectangle(final_result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    lbl_pos = (x, y-5)
                cv2.putText(final_result_img, obj['label'], lbl_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.addWeighted(overlay_mask, 0.4, final_result_img, 0.6, 0, final_result_img)
        final_result_img = final_result_img[0:orig_h, 0:orig_w]
        status_msg = f"Hoàn tất. Tìm thấy {count} tàu."

    except Exception as e:
        print(f"❌ Lỗi Runtime: {e}")
        return None, f"Lỗi: {str(e)}"

    result_filename = f"{name}_{model_id}_result{ext}"
    cv2.imencode(ext, final_result_img)[1].tofile(os.path.join(save_dir, result_filename))

    return {
        'result': result_filename,
        'enhanced': enhanced_filename,
        'mask': mask_filename,
        'detail': detail_filename
    }, status_msg


# ==========================================
# 5. COMPARE MODELS
# ==========================================
def get_model_style(model_id):
    if 'yolo' in model_id: return {'bgr': (0,0,255,255), 'css': 'rgba(255, 0, 0, 1)'}
    elif 'r3det' in model_id: return {'bgr': (0,255,0,255), 'css': 'rgba(0, 255, 0, 1)'}
    elif 'oriented_rcnn' in model_id: return {'bgr': (255,0,0,255), 'css': 'rgba(0, 0, 255, 1)'}
    elif 'faster_rcnn' in model_id: return {'bgr': (0,255,255,255), 'css': 'rgba(255, 255, 0, 1)'}
    elif 'mask_rcnn' in model_id: return {'bgr': (255,0,255,255), 'css': 'rgba(255, 0, 255, 1)'}
    else: return {'bgr': (255,255,255,255), 'css': 'rgba(255, 255, 255, 1)'}

def compare_models_inference(image_path, conf_threshold=0.3, use_land_mask=False, use_advanced_proc=False):
    patches_data, display_img, enhanced_img, coastal_mask, detail_vis, original_shape, error = \
        preprocessing.preprocess_image_for_inference(image_path, use_advanced_proc=use_advanced_proc)
    if error: return None, None, f"Lỗi: {error}"

    orig_h, orig_w = original_shape
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    save_dir = os.path.join(BASE_DIR, 'static', 'uploads')

    proc_outputs = {}
    if enhanced_img is not None:
        f = f"{name}_enhanced{ext}"; cv2.imencode(ext, enhanced_img[0:orig_h, 0:orig_w])[1].tofile(os.path.join(save_dir, f)); proc_outputs['enhanced']=f
    if coastal_mask is not None:
        f = f"{name}_mask{ext}"; cv2.imencode(ext, coastal_mask[0:orig_h, 0:orig_w])[1].tofile(os.path.join(save_dir, f)); proc_outputs['mask']=f
    if detail_vis is not None and use_advanced_proc:
        f = f"{name}_detail{ext}"; cv2.imencode(ext, detail_vis[0:orig_h, 0:orig_w])[1].tofile(os.path.join(save_dir, f)); proc_outputs['detail']=f

    models_to_compare = ['yolo', 'r3det', 'oriented_rcnn', 'faster_rcnn', 'mask_rcnn']
    results = []

    for model_id in models_to_compare:
        model_data = get_model(model_id)
        if not model_data: continue
        model = model_data['model']; m_type = model_data['type']
        all_detections = []
        start_time = time.time()
        try:
            for patch_info in patches_data:
                img_input = patch_info['img']; off_x, off_y = patch_info['offset']
                if m_type == 'mmrotate':
                    res = inference_detector(model, img_input)
                    objs = process_mmrotate_result(res, off_x, off_y, score_thr=conf_threshold)
                    all_detections.extend(objs)
                elif m_type == 'yolo':
                    img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
                    res = model(img_rgb, imgsz=1024, verbose=False, device=DEVICE_NAME)
                    objs = process_yolo_result(res, off_x, off_y, score_thr=conf_threshold)
                    all_detections.extend(objs)
                elif m_type == 'torchvision' or m_type == 'mask_rcnn_torch':
                    img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
                    img_tensor = torchvision.transforms.functional.to_tensor(img_rgb).unsqueeze(0).to(model_data['device'])
                    with torch.no_grad(): pred = model(img_tensor)
                    objs = process_torchvision_result(pred, off_x, off_y, score_thr=conf_threshold)
                    all_detections.extend(objs)

            final_objects = simple_nms_merge(all_detections, iou_thr=0.2)
            end_time = time.time()
            inference_time = round(end_time - start_time, 2)

            h_pad, w_pad = display_img.shape[:2]
            overlay_box = np.zeros((h_pad, w_pad, 4), dtype=np.uint8)
            overlay_label = np.zeros((h_pad, w_pad, 4), dtype=np.uint8)
            style = get_model_style(model_id); color_bgr = style['bgr']
            count = 0; total_conf = 0; active_mask = coastal_mask if use_land_mask else None

            for obj in final_objects:
                if obj['type'] == 'poly':
                    M = cv2.moments(obj['box'])
                    cx, cy = (int(M['m10']/M['m00']), int(M['m01']/M['m00'])) if M['m00']!=0 else (obj['box'][0][0], obj['box'][0][1])
                else:
                    x, y, w, h = obj['box']; cx, cy = x + w//2, y + h//2

                if is_valid_location((cx, cy), active_mask):
                    count += 1; total_conf += obj['score']
                    if obj['type'] == 'poly':
                        cv2.polylines(overlay_box, [obj['box']], True, color_bgr, 2)
                        lbl_pos = tuple(obj['box'][0])
                    else:
                        x, y, w, h = obj['box']
                        cv2.rectangle(overlay_box, (x, y), (x+w, y+h), color_bgr, 2)
                        lbl_pos = (x, y-5)
                    
                    if obj.get('has_mask', False):
                        cv2.drawContours(overlay_box, obj['contours'], -1, (255, 0, 255, 100), -1)

                    cv2.putText(overlay_label, obj['label'], lbl_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0,255), 3)
                    cv2.putText(overlay_label, obj['label'], lbl_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255,255), 2)

            file_box = f"{name}_{model_id}_box.png"; file_label = f"{name}_{model_id}_label.png"
            cv2.imwrite(os.path.join(save_dir, file_box), overlay_box[0:orig_h, 0:orig_w])
            cv2.imwrite(os.path.join(save_dir, file_label), overlay_label[0:orig_h, 0:orig_w])

            avg_conf = round((total_conf/count*100),1) if count > 0 else 0
            results.append({'id': model_id, 'name': model_data.get('name', model_id), 'img_box': file_box, 'img_label': file_label, 'count': count, 'time': inference_time, 'conf': avg_conf, 'color': style['css']})
        except Exception as e:
            print(f"❌ Lỗi {model_id}: {e}"); continue

    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
    return results, proc_outputs


# ==========================================
# 6. VIDEO PROCESSING & TRACKING
# ==========================================
def predict_video_frame(model_id, frame, conf_threshold=0.3, tracker=None):
    model_data = get_model(model_id)
    if not model_data: return frame, []
    model = model_data['model']; m_type = model_data['type']
    detected_objects = []
    rects = []

    try:
        raw_objects = []
        if m_type == 'mmrotate':
            res = inference_detector(model, frame)
            raw_objects = process_mmrotate_result(res, 0, 0, score_thr=conf_threshold)
        elif m_type == 'yolo':
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = model(img_rgb, verbose=False, device=DEVICE_NAME)
            raw_objects = process_yolo_result(res, 0, 0, score_thr=conf_threshold)
        elif m_type == 'torchvision' or m_type == 'mask_rcnn_torch':
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = torchvision.transforms.functional.to_tensor(img_rgb).unsqueeze(0).to(model_data['device'])
            with torch.no_grad(): pred = model(img_tensor)
            raw_objects = process_torchvision_result(pred, 0, 0, score_thr=conf_threshold)

        for obj in raw_objects:
            if obj['type'] == 'poly':
                box_pts = np.array(obj['box'], dtype=np.int32)
                x, y, w, h = cv2.boundingRect(box_pts)
                rects.append((x, y, x + w, y + h))
            else:
                x, y, w, h = [int(v) for v in obj['box']]
                rects.append((x, y, x + w, y + h))
            detected_objects.append(obj)

        if tracker is not None:
            objects_dict, trails_dict = tracker.update(rects)
        else:
            objects_dict, trails_dict = {}, {}

        color = (0, 255, 0); text_color = (255, 255, 255)
        if 'yolo' in model_id: color = (0, 0, 255)
        elif 'oriented' in model_id: color = (255, 0, 0)
        elif 'faster' in model_id: color = (0, 255, 255); text_color = (0, 0, 0)
        elif 'mask_rcnn' in model_id: color = (255, 0, 255)

        for obj in detected_objects:
            score = obj.get('score', 0); label_conf = f"{score:.2f}"
            if obj['type'] == 'poly':
                cv2.polylines(frame, [np.array(obj['box'], dtype=np.int32)], True, color, 2)
                lbl_x, lbl_y = int(obj['box'][0][0]), int(obj['box'][0][1])
            else:
                x, y, w, h = [int(v) for v in obj['box']]
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                lbl_x, lbl_y = x, y-5

            (w_text, h_text), _ = cv2.getTextSize(label_conf, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (lbl_x, lbl_y - h_text - 4), (lbl_x + w_text, lbl_y), color, -1)
            cv2.putText(frame, label_conf, (lbl_x, lbl_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        for (objectID, centroid) in objects_dict.items():
            text_id = f"ID {objectID}"
            cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)
            cv2.putText(frame, text_id, (centroid[0] + 10, centroid[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if objectID in trails_dict:
                pts = trails_dict[objectID]
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], color, 1)

        return frame, detected_objects
    except Exception as e:
        print(f"❌ Lỗi Video Tracking: {e}"); return frame, []
