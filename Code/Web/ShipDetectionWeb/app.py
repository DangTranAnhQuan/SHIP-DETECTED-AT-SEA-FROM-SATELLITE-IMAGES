import os
import time
import json
import cv2
import gc
import torch
import sys
from flask import Flask, render_template, request, redirect, url_for, abort, Response
from werkzeug.utils import secure_filename

# ==========================================
# 🔴 [COLAB] SETUP PYNGROK
# ==========================================
try:
    from pyngrok import ngrok
except ImportError:
    print("⚠️ Chưa cài đặt pyngrok! Vui lòng chạy: !pip install pyngrok")
    sys.exit(1)

# Import module của bạn
from inference import predict_image, compare_models_inference, predict_video_frame
from tracker import CentroidTracker

app = Flask(__name__)

# Cấu hình thư mục upload
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tạo thư mục nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- BIẾN TOÀN CỤC CHO VIDEO ---
VIDEO_CONFIG = {
    'source': None,
    'model_id': None,
    'conf': 0.4,
    'running': False
}

CURRENT_STATS = {
    'fps': 0,
    'ship_count': 0,
    'avg_conf': 0,
    'is_active': False
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_data(filename):
    path = os.path.join('data', filename)
    # [COLAB FIX] Tạo file giả lập nếu chưa có để tránh lỗi crash app
    if not os.path.exists(path):
        print(f"⚠️ Cảnh báo: Không tìm thấy {path}. Đang tạo data mẫu...")
        os.makedirs('data', exist_ok=True)
        dummy_data = {}
        if 'models_info' in filename:
            dummy_data = {
                "yolo_v8": {"name": "YOLOv8 Ship", "type": "yolo", "weights_file": "weights/yolov8_best.pt"},
                "faster_rcnn": {"name": "Faster R-CNN", "type": "torchvision", "weights_file": "weights/faster_rcnn.pth"}
            }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f)
            
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ==========================================
# CÁC ROUTE CƠ BẢN
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dataset')
def dataset():
    info = load_data('dataset_info.json')
    return render_template('dataset.html', info=info)

@app.route('/models')
def models_list():
    models = load_data('models_info.json')
    return render_template('models_list.html', models=models)

@app.route('/model/<model_id>')
def model_detail(model_id):
    models = load_data('models_info.json')
    if model_id not in models:
        abort(404)
    return render_template('model_detail.html', model=models[model_id])

# ==========================================
# CHỨC NĂNG DEMO & COMPARE & VIDEO (GIỮ NGUYÊN)
# ==========================================
@app.route('/demo', methods=['GET', 'POST'])
def demo():
    models = load_data('models_info.json')
    default_conf = 0.4
    if request.method == 'POST':
        if 'file' not in request.files: return redirect(request.url)
        file = request.files['file']
        if file.filename == '': return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            selected_model = request.form.get('model_select')
            try: conf_threshold = float(request.form.get('conf_threshold', default_conf))
            except: conf_threshold = default_conf
            use_land_mask = 'use_land_mask' in request.form
            use_advanced_proc = 'use_advanced_proc' in request.form
            
            outputs, status = predict_image(filepath, selected_model, conf_threshold=conf_threshold, use_land_mask=use_land_mask, use_advanced_proc=use_advanced_proc)
            if outputs is None: return render_template('demo.html', models=models, status=status)
            return render_template('demo.html', models=models, original_img=filename, result_img=outputs['result'], enhanced_img=outputs['enhanced'], mask_img=outputs['mask'], detail_img=outputs.get('detail'), status=status, selected_model=selected_model, current_conf=conf_threshold, current_mask=use_land_mask, current_advanced=use_advanced_proc)
    return render_template('demo.html', models=models, current_conf=default_conf)

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    default_conf = 0.3
    if request.method == 'POST':
        filepath, filename = None, None
        file_obj = request.files.get('file')
        if file_obj and file_obj.filename != '' and allowed_file(file_obj.filename):
            timestamp = int(time.time())
            original_name = secure_filename(file_obj.filename)
            filename = f"{timestamp}_{original_name}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file_obj.save(filepath)
        elif request.form.get('previous_file'):
            filename = request.form.get('previous_file')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath): return redirect(request.url)
        else: return redirect(request.url)

        try: conf_threshold = float(request.form.get('conf_threshold', default_conf))
        except: conf_threshold = default_conf
        use_land_mask = 'use_land_mask' in request.form
        use_advanced_proc = 'use_advanced_proc' in request.form
        
        results, proc_outputs = compare_models_inference(filepath, conf_threshold=conf_threshold, use_land_mask=use_land_mask, use_advanced_proc=use_advanced_proc)
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
        if results is None: return render_template('compare.html', error=proc_outputs)
        return render_template('compare.html', original_img=filename, results=results, enhanced_img=proc_outputs.get('enhanced'), mask_img=proc_outputs.get('mask'), detail_img=proc_outputs.get('detail'), current_conf=conf_threshold, current_mask=use_land_mask, current_advanced=use_advanced_proc)
    return render_template('compare.html', results=None)

@app.route('/video_ui')
def video_ui():
    models = load_data('models_info.json')
    return render_template('video.html', models=models)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files: return redirect(request.url)
    file = request.files['file']
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        VIDEO_CONFIG['source'] = filepath
        VIDEO_CONFIG['model_id'] = request.form.get('model_select')
        try: VIDEO_CONFIG['conf'] = float(request.form.get('conf_threshold', 0.4))
        except: VIDEO_CONFIG['conf'] = 0.4
        VIDEO_CONFIG['running'] = True
        models = load_data('models_info.json')
        return render_template('video.html', models=models, ready=True, current_model=VIDEO_CONFIG['model_id'])
    return redirect(url_for('video_ui'))

def generate_video_frames():
    source = VIDEO_CONFIG['source']
    model_id = VIDEO_CONFIG['model_id']
    conf = VIDEO_CONFIG['conf']
    if not source or not model_id: return
    cap = cv2.VideoCapture(source)
    tracker = CentroidTracker(maxDisappeared=50)
    frame_count = 0
    skip_rate = 2
    last_frame_processed = None
    CURRENT_STATS['is_active'] = True
    while cap.isOpened():
        loop_start = time.time()
        success, frame = cap.read()
        if not success: break
        frame_count += 1
        if frame_count % skip_rate == 0:
            processed_frame, objects = predict_video_frame(model_id, frame, conf, tracker=tracker)
            num_ships = len(objects)
            avg_score = sum([obj.get('score', 0) for obj in objects]) / num_ships if num_ships > 0 else 0.0
            CURRENT_STATS['ship_count'] = num_ships
            CURRENT_STATS['avg_conf'] = round(avg_score * 100, 1)
            last_frame_processed = processed_frame
        else:
            processed_frame = last_frame_processed if last_frame_processed is not None else frame
        
        process_time = time.time() - loop_start
        if process_time > 0: CURRENT_STATS['fps'] = int(1.0 / process_time)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    CURRENT_STATS['is_active'] = False
    cap.release()

@app.route('/video_data_feed')
def video_data_feed():
    def generate():
        while True:
            if CURRENT_STATS['is_active']:
                yield f"data:{json.dumps(CURRENT_STATS)}\n\n"
            else:
                yield f"data:{json.dumps({'is_active': False})}\n\n"
            time.sleep(0.5)
    return Response(generate(), mimetype='text/event-stream')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ==========================================
# 🔴 [COLAB] CHẠY APP QUA NGROK
# ==========================================
if __name__ == '__main__':
    port = 5000
    
    # 1. Kill tiến trình cũ (nếu restart cell)
    print("🔄 Đang dọn dẹp port cũ...")
    ngrok.kill()
    
    # 2. (Tuỳ chọn) Set Authtoken nếu có
    ngrok.set_auth_token("37C4vzs4LjcmG0Qgy0swEoQqGIB_212L5E8ybtyDGZS9or7kn") 

    # 3. Mở tunnel
    public_url = ngrok.connect(port).public_url
    print(f"\n🚀 WEBSITE ĐANG CHẠY TẠI: {public_url} \n")
    print("⚠️ Lưu ý: Nếu trang web báo lỗi 'ngrok errors', hãy đăng ký tài khoản ngrok miễn phí và thêm token vào code.")
    
    # 4. Chạy Flask (tắt debug reloader để tránh lỗi trên Colab)
    app.run(port=port, debug=False)
