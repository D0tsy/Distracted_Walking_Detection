import cv2
import os
import time
import numpy as np
import torch
from collections import deque
import supervision as sv
from ultralytics import YOLO
from rfdetr import RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES

# ============================================================
# CONFIGURATION
# ============================================================

PHONE_CONF_THRESHOLD = 0.30
HEAD_TILT_THREASHOLD = 0.55 #higher = less strict (previously 0.55)

# --- Debug mode ---
# True  : show all overlays (skeleton, keypoints, phone boxes, expansion zones, debug text)
# False : clean output — only person bounding box (green/red) + stats
DEBUG_MODE = False

# --- Temporal streak filter ---
STREAK_REQUIRED  = 2
DROPOUT_ALLOWED  = 3

# --- Hybrid matcher ---
IOU_MATCH_THRESHOLD = 0.05
MAX_MATCH_DISTANCE  = 0.08  # fraction of frame diagonal

# --- Velocity-aware dropout ---
FAST_MOVE_THRESHOLD = 0.03  # fraction of frame diagonal

# --- Violation debounce ---
VIOLATION_DEBOUNCE_FRAMES = 1
VIOLATION_HOLD_FRAMES     = 10

# --- Walking detection ---
BASE_WALKING_THRESHOLD = 3    # pixels per frame at z_scale=1.0
WALKING_HISTORY_FRAMES = 10


VIDEO_SOURCE = 'vid1.mp4'   # 0 = default laptop webcam. Set to a filename string (e.g. 'Vid23.mp4') to read a video file instead.

# --- Model Image Size Resolution ---
POSE_IMGSZ = 640
PHONE_RES  = 864  

RFDETR_CHECKPOINT = None  # e.g. "weights/rfdetr_phone.pth"

# ============================================================
# DEVICE SETUP
# ============================================================

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
if CUDA_AVAILABLE:
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available — falling back to CPU. Check your torch/CUDA install.")

# ============================================================
# VIOLATION LOGIC
# ============================================================

def expand_phone_bboxes(phone_bboxes, expansion=100):
    expanded = []
    for bbox in phone_bboxes:
        x1, y1, x2, y2 = bbox
        expanded.append([x1 - expansion, y1 - expansion,
                         x2 + expansion, y2 + expansion])
    return expanded

def is_point_in_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def estimate_depth_scale(left_shoulder, right_shoulder, base_shoulder_width=150):
    if left_shoulder is None or right_shoulder is None:
        return 1.0
    current_width = abs(left_shoulder[0] - right_shoulder[0])
    if current_width < 10:
        return 1.0
    return max(0.5, min(current_width / base_shoulder_width, 3.0))

def check_violation_for_person(kp, phone_bboxes, person_bbox, person_id=0, avg_displacement=0.0):
    def is_valid(idx):
        return len(kp) > idx and kp[idx][2] > 0.5

    nose           = kp[0]  if is_valid(0)  else None
    left_eye       = kp[1]  if is_valid(1)  else None
    right_eye      = kp[2]  if is_valid(2)  else None
    left_ear       = kp[3]  if is_valid(3)  else None
    right_ear      = kp[4]  if is_valid(4)  else None
    left_shoulder  = kp[5]  if is_valid(5)  else None
    right_shoulder = kp[6]  if is_valid(6)  else None
    left_elbow     = kp[7]  if is_valid(7)  else None
    right_elbow    = kp[8]  if is_valid(8)  else None
    left_wrist     = kp[9]  if is_valid(9)  else None
    right_wrist    = kp[10] if is_valid(10) else None

    z_scale              = estimate_depth_scale(left_shoulder, right_shoulder)
    dynamic_expansion    = int(25 * z_scale)
    dynamic_arm_drop     = int(40 * z_scale)
    phone_expansion      = int(70 * z_scale)
    expanded_phone_bboxes = expand_phone_bboxes(phone_bboxes, phone_expansion)

    walking_threshold = BASE_WALKING_THRESHOLD * z_scale
    is_walking        = avg_displacement > walking_threshold

    highlight_points = []
    for pt, label in [(nose, 'NOSE'), (left_shoulder, 'LSH'), (right_shoulder, 'RSH'),
                      (left_eye, 'LEYE'), (right_eye, 'REYE'), (left_ear, 'LEAR'), (right_ear, 'REAR'),
                      (left_elbow, 'LELBOW'), (right_elbow, 'RELBOW'),
                      (left_wrist, 'LWRIST'), (right_wrist, 'RWRIST')]:
        if pt is not None:
            highlight_points.append((int(pt[0]), int(pt[1]), f'ID{person_id}_{label}', dynamic_expansion))

    # Head angle logic
    eyes_facing_down  = False
    head_debug_reason = "Looking Straight/Unknown"
    head_drop_ratio    = None   # raw nose-drop ratio, kept for debug display even when not a violation

    if nose is not None:
        shoulder_y = reference_length = None

        if left_shoulder is not None and right_shoulder is not None:
            shoulder_y       = (left_shoulder[1] + right_shoulder[1]) / 2
            reference_length = abs(left_shoulder[0] - right_shoulder[0])
        elif left_shoulder is not None and left_ear is not None:
            shoulder_y       = left_shoulder[1]
            reference_length = abs(left_shoulder[1] - left_ear[1]) * 1.5
        elif right_shoulder is not None and right_ear is not None:
            shoulder_y       = right_shoulder[1]
            reference_length = abs(right_shoulder[1] - right_ear[1]) * 1.5

        if shoulder_y is not None and reference_length is not None and reference_length > 10:
            head_drop_ratio = (shoulder_y - nose[1]) / reference_length
            if head_drop_ratio < HEAD_TILT_THREASHOLD:
                eyes_facing_down  = True
                head_debug_reason = f"Nose dropped (Ratio: {head_drop_ratio:.2f})"

    if not eyes_facing_down:
        if left_ear is not None and left_eye is not None and left_ear[1] < left_eye[1]:
            eyes_facing_down  = True
            head_debug_reason = "Left ear above eye"
        elif right_ear is not None and right_eye is not None and right_ear[1] < right_eye[1]:
            eyes_facing_down  = True
            head_debug_reason = "Right ear above eye"

    face_visible      = any(x is not None for x in [nose, left_eye, right_eye, left_ear, right_ear])
    shoulders_visible = left_shoulder is not None or right_shoulder is not None
    facing_away       = shoulders_visible and not face_visible

    if eyes_facing_down:
        head_state = "down"
    elif facing_away:
        head_state = "unknown"
    else:
        head_state = "up"

    # Arm and phone logic
    has_left_arm  = left_elbow  is not None and left_wrist  is not None
    has_right_arm = right_elbow is not None and right_wrist is not None

    left_violation = right_violation = False
    left_phone_near = left_arm_valid = False
    right_phone_near = right_arm_valid = False

    if has_left_arm:
        left_arm_valid  = left_wrist[1] <= (left_elbow[1] + dynamic_arm_drop)
        left_phone_near = any(is_point_in_bbox([left_wrist[0], left_wrist[1]], b)
                              for b in expanded_phone_bboxes)

    if has_right_arm:
        right_arm_valid  = right_wrist[1] <= (right_elbow[1] + dynamic_arm_drop)
        right_phone_near = any(is_point_in_bbox([right_wrist[0], right_wrist[1]], b)
                               for b in expanded_phone_bboxes)

    if is_walking:
        left_violation  = left_phone_near  and left_arm_valid and eyes_facing_down
        right_violation = right_phone_near and right_arm_valid and eyes_facing_down

    is_violation = left_violation or right_violation

    if left_violation and right_violation:
        violation_side = "BOTH"
    elif left_violation:
        violation_side = "LEFT"
    elif right_violation:
        violation_side = "RIGHT"
    else:
        violation_side = None

    debug_info = {
        'person_id':         person_id,
        'person_bbox':       person_bbox,
        'is_violation':      is_violation,
        'head_reason':       head_debug_reason,
        'head_state':        head_state,
        'head_drop_ratio':   head_drop_ratio,
        'z_scale':           z_scale,
        'dynamic_expansion': dynamic_expansion,
        'phone_expansion':   phone_expansion,
        'eyes_facing_down':  eyes_facing_down,
        'facing_away':       facing_away,
        'left_phone_near':   left_phone_near,
        'left_arm_valid':    left_arm_valid,
        'right_phone_near':  right_phone_near,
        'right_arm_valid':   right_arm_valid,
        'is_walking':        is_walking,
        'avg_displacement':  avg_displacement,
        'walking_threshold': walking_threshold,
    }

    return is_violation, violation_side, highlight_points, debug_info

def check_violation_all_persons(pose_results, phone_bboxes, person_displacement=None):
    pose_keypoints = pose_results[0].keypoints.data.cpu().numpy() if pose_results[0].keypoints else None

    person_bboxes = track_ids = None
    if pose_results[0].boxes is not None and len(pose_results[0].boxes) > 0:
        person_bboxes = pose_results[0].boxes.xyxy.cpu().numpy()
        if pose_results[0].boxes.id is not None:
            track_ids = pose_results[0].boxes.id.int().cpu().tolist()

    if pose_keypoints is None or len(pose_keypoints) == 0:
        return False, [], [], {}

    if person_displacement is None:
        person_displacement = {}

    all_violations       = []
    all_highlight_points = []
    any_violation        = False
    all_person_data      = {}

    for i in range(len(pose_keypoints)):
        kp          = pose_keypoints[i]
        person_bbox = person_bboxes[i] if person_bboxes is not None and i < len(person_bboxes) else None
        person_id   = track_ids[i] if track_ids is not None and i < len(track_ids) else f"Temp_{i}"
        avg_disp    = person_displacement.get(person_id, 0.0)

        is_violation, violation_side, highlight_points, debug_info = check_violation_for_person(
            kp, phone_bboxes, person_bbox, person_id=person_id, avg_displacement=avg_disp
        )

        all_highlight_points.extend(highlight_points)
        all_person_data[person_id] = debug_info

        if is_violation:
            any_violation = True
            all_violations.append({'person_id': person_id, 'side': violation_side, 'debug_info': debug_info})

    return any_violation, all_violations, all_highlight_points, all_person_data

# ============================================================
# MAIN
# ============================================================

def main():
    global DEBUG_MODE
    base_dir = os.path.dirname(os.path.abspath(__file__))

    violations_dir = os.path.join(base_dir, 'violations')
    os.makedirs(violations_dir, exist_ok=True)

    # --- Load models ---
    print("Loading YOLO pose model...")
    pose_model = YOLO('yolo26l-pose.pt')
    pose_model.to(DEVICE)
    POSE_DEVICE = DEVICE
    print(f"YOLO pose running on: {POSE_DEVICE.upper()}")

    print("Loading RF-DETR phone detection model (PyTorch/CUDA)...")
    rfdetr_kwargs = {"resolution": PHONE_RES, "device": DEVICE}
    if RFDETR_CHECKPOINT:
        rfdetr_kwargs["pretrain_weights"] = RFDETR_CHECKPOINT
    phone_model = RFDETRLarge(**rfdetr_kwargs)
    if DEVICE == "cuda":
        phone_model.optimize_for_inference()
    print(f"RF-DETR running on: {DEVICE.upper()}")
    print("Models loaded.")

    # --- Open video ---
    video_path = os.path.join(base_dir, VIDEO_SOURCE) if isinstance(VIDEO_SOURCE, str) else VIDEO_SOURCE
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        return

    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        print("Warning: source reported FPS <= 0 (common on some RTSP/CCTV streams) — defaulting to 25")
        fps = 25
    print(f"Video: {width}x{height} @ {fps}fps")

    # --- Output writer ---
    output_path = os.path.join(base_dir, 'output.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- Display window ---
    window_name = 'Distracted Walking Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    max_display_height = 800
    if height > max_display_height:
        scale = max_display_height / height
        cv2.resizeWindow(window_name, int(width * scale), max_display_height)
    else:
        cv2.resizeWindow(window_name, width, height)

    # --- State ---
    frame_count          = 0
    prev_frame_time      = 0
    violation_debounce   = {}
    violation_hold       = {}
    violation_side_cache = {}
    walking_history      = {}

    # violation summary — person_id -> total frames confirmed violating
    violation_summary    = {}

    # --- Timing diagnostics ---
    pose_times  = deque(maxlen=60)
    phone_times = deque(maxlen=60)
    TIMING_LOG_EVERY = 30

    # --- RF-DETR instance tracker ---
    instances = {}
    next_id   = [0]

    frame_diag       = (width ** 2 + height ** 2) ** 0.5
    max_dist_pixels  = MAX_MATCH_DISTANCE * frame_diag
    fast_move_pixels = FAST_MOVE_THRESHOLD * frame_diag

    def box_center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def center_distance(boxA, boxB):
        cx1, cy1 = box_center(boxA)
        cx2, cy2 = box_center(boxB)
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

    def compute_iou(boxA, boxB):
        ax1, ay1, ax2, ay2 = boxA
        bx1, by1, bx2, by2 = boxB
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)

    def is_match(boxA, boxB):
        return (compute_iou(boxA, boxB) >= IOU_MATCH_THRESHOLD or
                center_distance(boxA, boxB) <= max_dist_pixels)

    print("Running... Press 'q' to quit, 'd' to toggle debug mode.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --------------------------------------------------------
        # 1. POSE DETECTION  (YOLO, CUDA)
        # --------------------------------------------------------
        _t0 = time.time()
        pose_results = pose_model.track(frame, persist=True, tracker="tracktrack.yaml",
                                        verbose=False, imgsz=POSE_IMGSZ, device=POSE_DEVICE,
                                        half=(DEVICE == "cuda"))
        pose_times.append(time.time() - _t0)

        # --------------------------------------------------------
        # 2. PHONE DETECTION  (RF-DETR, PyTorch/CUDA + instance tracker)
        # --------------------------------------------------------
        _t0 = time.time()
        rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rf_results = phone_model.predict(rgb_frame, threshold=PHONE_CONF_THRESHOLD)
        phone_times.append(time.time() - _t0)

        # Filter down to the "cell phone" class only (model may be trained/
        # loaded with the full COCO label set).
        raw_detections = []
        if rf_results is not None and len(rf_results) > 0:
            for bbox, cls_id, conf in zip(rf_results.xyxy, rf_results.class_id, rf_results.confidence):
                if COCO_CLASSES[int(cls_id)] == 'cell phone':
                    raw_detections.append((bbox, float(conf)))

        matched_raw_indices = set()
        for inst in instances.values():
            best_score = best_idx = None
            for i, (bbox, conf) in enumerate(raw_detections):
                if i in matched_raw_indices or not is_match(inst['bbox'], bbox):
                    continue
                score = (compute_iou(inst['bbox'], bbox), -center_distance(inst['bbox'], bbox))
                if best_score is None or score > best_score:
                    best_score, best_idx = score, i

            if best_idx is not None:
                bbox, conf           = raw_detections[best_idx]
                inst['velocity']     = center_distance(inst['bbox'], bbox)
                inst['bbox']         = bbox
                inst['conf']         = conf
                inst['streak']      += 1
                inst['dropout']      = 0
                if inst['streak'] >= STREAK_REQUIRED:
                    inst['active']   = True
                matched_raw_indices.add(best_idx)
            else:
                inst['streak'] = 0
                if inst['active']:
                    effective_dropout = 0 if inst.get('velocity', 0) > fast_move_pixels else DROPOUT_ALLOWED
                    inst['dropout'] += 1
                    if inst['dropout'] > effective_dropout:
                        inst['active'] = inst['dropout'] = False

        for i, (bbox, conf) in enumerate(raw_detections):
            if i not in matched_raw_indices:
                instances[next_id[0]] = {
                    'bbox': bbox, 'conf': conf,
                    'streak': 1, 'dropout': 0,
                    'active': False, 'velocity': 0,
                }
                next_id[0] += 1

        stale = [tid for tid, inst in instances.items()
                 if not inst['active'] and inst['streak'] == 0 and inst['dropout'] == 0]
        for tid in stale:
            del instances[tid]

        phone_bboxes = [inst['bbox'] for inst in instances.values() if inst['active']]

        # --------------------------------------------------------
        # 3. WALKING DETECTION
        # --------------------------------------------------------
        all_person_bboxes = track_ids = None
        if pose_results[0].boxes is not None and len(pose_results[0].boxes) > 0:
            all_person_bboxes = pose_results[0].boxes.xyxy.cpu().numpy()
            if pose_results[0].boxes.id is not None:
                track_ids = pose_results[0].boxes.id.int().cpu().tolist()

        person_displacement = {}
        if all_person_bboxes is not None:
            active_ids     = set()
            pose_keypoints = pose_results[0].keypoints.data.cpu().numpy() \
                             if pose_results[0].keypoints else None

            for i, bbox in enumerate(all_person_bboxes):
                person_id = track_ids[i] if track_ids is not None and i < len(track_ids) else f"Temp_{i}"
                active_ids.add(person_id)

                cx = cy = None
                if pose_keypoints is not None and i < len(pose_keypoints):
                    kp   = pose_keypoints[i]
                    l_sh = kp[5] if kp[5][2] > 0.5 else None
                    r_sh = kp[6] if kp[6][2] > 0.5 else None
                    if l_sh is not None and r_sh is not None:
                        cx, cy = (l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2
                    elif l_sh is not None:
                        cx, cy = l_sh[0], l_sh[1]
                    elif r_sh is not None:
                        cx, cy = r_sh[0], r_sh[1]

                if cx is None:
                    x1, y1, x2, y2 = bbox
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                if person_id not in walking_history:
                    walking_history[person_id] = deque(maxlen=WALKING_HISTORY_FRAMES)
                walking_history[person_id].append((cx, cy))

                history = walking_history[person_id]
                if len(history) >= 2:
                    displacements = [
                        ((history[j][0] - history[j-1][0]) ** 2 +
                         (history[j][1] - history[j-1][1]) ** 2) ** 0.5
                        for j in range(1, len(history))
                    ]
                    person_displacement[person_id] = sum(displacements) / len(displacements)
                else:
                    person_displacement[person_id] = 0.0

            for pid in [p for p in walking_history if p not in active_ids]:
                del walking_history[pid]

        # --------------------------------------------------------
        # 4. VIOLATION LOGIC
        # --------------------------------------------------------
        _, raw_violations_list, highlight_points, all_person_data = \
            check_violation_all_persons(pose_results, phone_bboxes,
                                        person_displacement=person_displacement)

        raw_violator_ids       = {v['person_id']: v for v in raw_violations_list}
        confirmed_violator_ids = set()

        if all_person_bboxes is not None:
            for i in range(len(all_person_bboxes)):
                person_id = track_ids[i] if track_ids is not None and i < len(track_ids) else f"Temp_{i}"

                if person_id in raw_violator_ids:
                    violation_debounce[person_id] = min(
                        VIOLATION_DEBOUNCE_FRAMES,
                        violation_debounce.get(person_id, 0) + 1
                    )
                    if violation_debounce[person_id] >= VIOLATION_DEBOUNCE_FRAMES:
                        violation_hold[person_id]       = VIOLATION_HOLD_FRAMES
                        violation_side_cache[person_id] = raw_violator_ids[person_id]['side']
                else:
                    violation_debounce[person_id] = 0
                    violation_hold[person_id] = max(0, violation_hold.get(person_id, 0) - 1)

                if violation_hold.get(person_id, 0) > 0:
                    confirmed_violator_ids.add(person_id)
                    violation_summary[person_id] = violation_summary.get(person_id, 0) + 1

        # --------------------------------------------------------
        # 5. DRAWING
        # --------------------------------------------------------
        if DEBUG_MODE:
            annotated_frame = pose_results[0].plot(line_width=2, font_size=0.5,
                                                    kpt_radius=3, kpt_line=True) # 2-3
        else:
            annotated_frame = frame.copy()

        if DEBUG_MODE:
            for inst in instances.values():
                if not inst['active']:
                    continue
                x1, y1, x2, y2 = map(int, inst['bbox'])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                tag = f"phone {inst['conf']:.2f}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 255, 255), -1)
                cv2.putText(annotated_frame, tag, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            for phone_bbox in phone_bboxes:
                if all_person_data:
                    max_exp = max(d['phone_expansion'] for d in all_person_data.values())
                    ex1, ey1, ex2, ey2 = map(int, expand_phone_bboxes([phone_bbox], max_exp)[0])
                    cv2.rectangle(annotated_frame, (ex1, ey1), (ex2, ey2), (0, 165, 255), 1)

        if all_person_bboxes is not None:
            for i, bbox in enumerate(all_person_bboxes):
                x1, y1, x2, y2 = map(int, bbox)
                person_id  = track_ids[i] if track_ids is not None and i < len(track_ids) else f"Temp_{i}"
                color      = (0, 0, 255) if person_id in confirmed_violator_ids else (0, 255, 0)
                label_text = f"ID:{person_id} - VIOLATOR" if person_id in confirmed_violator_ids else f"ID:{person_id}"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                cv2.putText(annotated_frame, label_text, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                #if DEBUG_MODE and person_id in all_person_data:
                if person_id in all_person_data:
                    data   = all_person_data[person_id]
                    h_state = data['head_state']
                    head  = "Y" if h_state == "down" else ("?" if h_state == "unknown" else "N")
                    lp    = "Y" if data['left_phone_near']  else "N"
                    la    = "Y" if data['left_arm_valid']   else "N"
                    rp    = "Y" if data['right_phone_near'] else "N"
                    ra    = "Y" if data['right_arm_valid']  else "N"
                    walk  = "Y" if data['is_walking']       else "N"
                    disp  = data['avg_displacement']
                    thr   = data['walking_threshold']
                    cv2.putText(annotated_frame,
                                f"H:{head} LP:{lp} LA:{la} RP:{rp} RA:{ra} W:{walk}({disp:.1f}/{thr:.1f})",
                                (x1 + 20, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    ratio_str = f"{data['head_drop_ratio']:.2f}" if data['head_drop_ratio'] is not None else "n/a"
                    cv2.putText(annotated_frame,
                                f"tiltRatio:{ratio_str}",
                                (x1 + 20, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

        if DEBUG_MODE:
            for x, y, label, dyn_tol in highlight_points:
                cv2.circle(annotated_frame, (x, y), 5, (0, 0, 255), -1)
                if 'WRIST' in label:
                    cv2.circle(annotated_frame, (x, y), dyn_tol, (255, 0, 255), 1)

        if confirmed_violator_ids and frame_count % fps == 0:
            for pid in confirmed_violator_ids:
                snap_path = os.path.join(
                    violations_dir,
                    f"violation_ID{pid}_{frame_count:06d}.jpg"
                )
                cv2.imwrite(snap_path, annotated_frame)

        # Stats bar
        new_frame_time  = time.time()
        process_fps     = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        num_tracked     = len(all_person_bboxes) if all_person_bboxes is not None else 0
        num_phones      = len(phone_bboxes)

        cv2.putText(annotated_frame,
                    f"Frame: {frame_count} | Persons: {num_tracked} | Phones: {num_phones} | FPS: {process_fps:.1f}",
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

        # Timing diagnostics
        if frame_count % TIMING_LOG_EVERY == 0 and pose_times and phone_times:
            avg_pose  = sum(pose_times)  / len(pose_times)  * 1000
            avg_phone = sum(phone_times) / len(phone_times) * 1000
            print(f"[Frame {frame_count}] Pose: {avg_pose:.1f}ms | Phone: {avg_phone:.1f}ms | "
                  f"Combined: {avg_pose + avg_phone:.1f}ms (~{1000/(avg_pose+avg_phone):.1f} FPS max)")

        out.write(annotated_frame)
        cv2.imshow(window_name, annotated_frame)

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            DEBUG_MODE = not DEBUG_MODE
            print(f"Debug mode: {'ON' if DEBUG_MODE else 'OFF'}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # --------------------------------------------------------
    # END-OF-RUN SUMMARY
    # --------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Frames processed : {frame_count}")
    print(f"Output saved to  : output.mp4")
    print(f"Snapshots saved  : {violations_dir}")
    if violation_summary:
        print(f"\nViolation summary (confirmed frames per person):")
        for pid, count in sorted(violation_summary.items(), key=lambda x: -x[1]):
            duration_sec = count / fps if fps > 0 else 0
            print(f"  ID {pid:>4} : {count} frames (~{duration_sec:.1f}s)")
    else:
        print("\nNo violations detected.")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()