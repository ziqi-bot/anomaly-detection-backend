

# import cv2
# import numpy as np
# import onnxruntime
# import time
# import os
# import glob

# _COLORS = (
#     np.array(
#         [
#             0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494,
#             0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
#             0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
#             1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000,
#             0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667,
#             0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
#             0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000,
#             1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
#             0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
#             0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
#             0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
#             0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
#             0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
#             0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
#             1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
#             1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.333,
#             0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000,
#             0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000,
#             0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000,
#             1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000,
#             0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000,
#             0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429,
#             0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714, 0.857, 0.857,
#             0.857, 0.000, 0.447, 0.741, 0.314, 0.717, 0.741, 0.50, 0.5, 0,
#         ]
#     ).astype(np.float32).reshape(-1, 3)
# )

# def vis(image, boxes, scores, cls_ids, conf=0.5, class_names=None, out_img=None, print_bbox=False):
#     width = image.shape[1]
#     height = image.shape[0]
#     sorted_bcs = sorted(zip(boxes, cls_ids, scores), key=lambda zipped: zipped[0][0])
#     for box, cls_id, score in sorted_bcs:
#         cls_id = int(cls_id)
#         if class_names is None:
#             label = cls_id
#         else:
#             label = class_names[cls_id]
#         if score < conf:
#             continue
#         x0 = int(box[0] * width)
#         y0 = int(box[1] * height)
#         x1 = int(box[2] * width)
#         y1 = int(box[3] * height)
#         color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
#         text = "{}: {:.2f}".format(label, score)
#         txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         txt_size = cv2.getTextSize(text, font, 0.6, 1)[0]
#         cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
#         txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
#         cv2.rectangle(image, (x0, y0 - int(1.5 * txt_size[1])), (x0 + txt_size[0] + 1, y0 - 1), txt_bk_color, -1)
#         cv2.putText(image, text, (x0, y0 - int(0.5 * txt_size[1])), font, 0.6, txt_color, thickness=1)
#         if print_bbox:
#             print(f"{label}: {score * 100.0:.0f}%\t(left_x: {x0}\ttop_y:  {y0}\twidth:  {x1 - x0}\theight:  {y1 - y0})")
#     if out_img:
#         print("save visualization to {}".format(out_img))
#         cv2.imwrite(out_img, image)
#     return image

# def nms(boxes, scores, nms_thr):
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]
#     areas = (x2 - x1) * (y2 - y1)
#     order = scores.argsort()[::-1]
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])
#         w = np.maximum(0.0, xx2 - xx1)
#         h = np.maximum(0.0, yy2 - yy1)
#         inter = w * h
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)
#         inds = np.where(ovr <= nms_thr)[0]
#         order = order[inds + 1]
#     return keep

# def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
#     final_dets = []
#     num_classes = scores.shape[1]
#     for cls_ind in range(num_classes):
#         cls_scores = scores[:, cls_ind]
#         valid_score_mask = cls_scores > score_thr
#         if valid_score_mask.sum() == 0:
#             continue
#         else:
#             valid_scores = cls_scores[valid_score_mask]
#             valid_boxes = boxes[valid_score_mask]
#             keep = nms(valid_boxes, valid_scores, nms_thr)
#             if len(keep) > 0:
#                 cls_inds = np.ones((len(keep), 1)) * cls_ind
#                 dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
#                 final_dets.append(dets)
#     if len(final_dets) == 0:
#         return None
#     return np.concatenate(final_dets, 0)

# def get_detections(predictions, score_thresh=0.5, nms_thresh=0.45):
#     boxes = predictions[:, :4]
#     scores = predictions[:, 4:]
#     dets = multiclass_nms_class_aware(boxes, scores, nms_thr=nms_thresh, score_thr=score_thresh)
#     if dets is not None:
#         final_boxes = dets[:, :4]
#         final_scores = dets[:, 4]
#         final_cls_inds = dets[:, 5]
#         return final_boxes, final_scores, final_cls_inds
#     else:
#         return None, None, None

# def read_names(names_path):
#     if names_path == "":
#         return None
#     class_names = []
#     with open(names_path, "r") as f:
#         for line in f:
#             class_names.append(line.strip())
#     return class_names

# def detect(session, image, score_thresh=0.1, nms_thresh=0.45, to_float16=False):
#     t1 = time.time()
#     IN_IMAGE_H = session.get_inputs()[0].shape[2]
#     IN_IMAGE_W = session.get_inputs()[0].shape[3]
#     resized = cv2.resize(image, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
#     img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#     img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)  # HWC to CHW
#     img_in /= 255.0
#     img_in = np.expand_dims(img_in, axis=0)
#     if to_float16:
#         img_in = img_in.astype(np.float16)

#     t2 = time.time()
#     input_name = session.get_inputs()[0].name
#     for _ in range(1):
#         _ = session.run(None, {input_name: img_in})

#     t3 = time.time()
#     outputs = session.run(None, {input_name: img_in})
#     outputs = [output.astype(np.float32) for output in outputs]

#     t4 = time.time()
#     final_boxes, final_scores, final_cls_inds = get_detections(outputs[0][0], score_thresh, nms_thresh)
#     t5 = time.time()

#     # print(f"Preprocessing : {t2 - t1:.4f}s")
#     # print(f"Inference     : {(t4 - t3) / 1:.4f}s")
#     # print(f"Postprocessing: {t5 - t4:.4f}s")
#     # print(f"Total         : {t2 - t1 + (t4 - t3) / 1 + t5 - t4:.4f}s")
#     return final_boxes, final_scores, final_cls_inds

# def process_latest_video():
#     upload_folder = 'uploads'
#     video_files = glob.glob(os.path.join(upload_folder, '*.mp4'))
#     if not video_files:
#         print("No video files found in upload folder.")
#         return
    
#     latest_video = max(video_files, key=os.path.getctime)
#     print(f"Processing video: {latest_video}")

#     score_thresh = 0.3
#     nms_thresh = 0.45
#     names_path = os.getenv('NAMES_PATH')
#     model_path = os.getenv('MODEL_PATH')

#     print(f"Model path: {model_path}")
#     print(f"Names path: {names_path}")

#     class_names = read_names(names_path)
#     session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])


#     cap = cv2.VideoCapture(latest_video)
#     frame_index = 0
#     save_dir = 'processed_frames'
#     os.makedirs(save_dir, exist_ok=True)

#     frame_size = None  # 初始化 frame_size

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame is None:
#             continue  # 如果 frame 是 None，则跳过本次循环

#         final_boxes, final_scores, final_cls_ids = detect(session, frame)
#         if final_boxes is not None:
#             frame = vis(frame, final_boxes, final_scores, final_cls_ids, class_names=class_names)
        
#         if frame_size is None:  # 设置 frame_size
#             frame_size = (frame.shape[1], frame.shape[0])
        
#         image_path = os.path.join(save_dir, f'frame_{frame_index}.jpg')
#         cv2.imwrite(image_path, frame)
#         print(f"Saved processed frame to {image_path}")
#         frame_index += 1

#     cap.release()

#     if frame_size is None:
#         print("No frames processed.")
#         return

#     # 合成视频
#     output_video_path = 'output/output_video.mp4'
#     out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, frame_size)

#     for i in range(frame_index):
#         frame_path = os.path.join(save_dir, f'frame_{i}.jpg')
#         frame = cv2.imread(frame_path)
#         out.write(frame)

#     out.release()
#     print(f"Processed video saved to {output_video_path}")

#     # 创建信号文件，通知后端处理已完成
#     signal_file_path = os.path.join('output', 'done.signal')
#     with open(signal_file_path, 'w') as f:
#         f.write('done')
#     print(f"Signal file created at {signal_file_path}")

# if __name__ == "__main__":
#     process_latest_video()























import cv2
import numpy as np
import onnxruntime
import time
import os
import glob

# Define colors for visualization
_COLORS = (
    np.array(
        [
            0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494,
            0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
            0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
            1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000,
            0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667,
            0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
            0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000,
            1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
            0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
            0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
            0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
            0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
            0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
            0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
            1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
            1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.333,
            0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000,
            0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000,
            0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000,
            1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000,
            0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000,
            0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429,
            0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714, 0.857, 0.857,
            0.857, 0.000, 0.447, 0.741, 0.314, 0.717, 0.741, 0.50, 0.5, 0,
        ]
    ).astype(np.float32).reshape(-1, 3)
)

def vis(image, boxes, scores, cls_ids, conf=0.5, class_names=None, out_img=None, print_bbox=False):
    width = image.shape[1]
    height = image.shape[0]
    sorted_bcs = sorted(zip(boxes, cls_ids, scores), key=lambda zipped: zipped[0][0])
    for box, cls_id, score in sorted_bcs:
        cls_id = int(cls_id)
        if class_names is None:
            label = cls_id
        else:
            label = class_names[cls_id]
        if score < conf:
            continue
        x0 = int(box[0] * width)
        y0 = int(box[1] * height)
        x1 = int(box[2] * width)
        y1 = int(box[3] * height)
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = "{}: {:.2f}".format(label, score)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.6, 1)[0]
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(image, (x0, y0 - int(1.5 * txt_size[1])), (x0 + txt_size[0] + 1, y0 - 1), txt_bk_color, -1)
        cv2.putText(image, text, (x0, y0 - int(0.5 * txt_size[1])), font, 0.6, txt_color, thickness=1)
        if print_bbox:
            print(f"{label}: {score * 100.0:.0f}%\t(left_x: {x0}\ttop_y:  {y0}\twidth:  {x1 - x0}\theight:  {y1 - y0})")
    if out_img:
        print("save visualization to {}".format(out_img))
        cv2.imwrite(out_img, image)
    return image

def nms(boxes, scores, nms_thr):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep

def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def get_detections(predictions, score_thresh=0.5, nms_thresh=0.45):
    boxes = predictions[:, :4]
    scores = predictions[:, 4:]
    dets = multiclass_nms_class_aware(boxes, scores, nms_thr=nms_thresh, score_thr=score_thresh)
    if dets is not None:
        final_boxes = dets[:, :4]
        final_scores = dets[:, 4]
        final_cls_inds = dets[:, 5]
        return final_boxes, final_scores, final_cls_inds
    else:
        return None, None, None

def read_names(names_path):
    if names_path == "":
        return None
    class_names = []
    with open(names_path, "r") as f:
        for line in f:
            class_names.append(line.strip())
    return class_names

def detect(session, image, score_thresh=0.1, nms_thresh=0.45, to_float16=False):
    t1 = time.time()
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]
    resized = cv2.resize(image, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)  # HWC to CHW
    img_in /= 255.0
    img_in = np.expand_dims(img_in, axis=0)
    if to_float16:
        img_in = img_in.astype(np.float16)

    t2 = time.time()
    input_name = session.get_inputs()[0].name
    for _ in range(1):
        _ = session.run(None, {input_name: img_in})

    t3 = time.time()
    outputs = session.run(None, {input_name: img_in})
    outputs = [output.astype(np.float32) for output in outputs]

    t4 = time.time()
    final_boxes, final_scores, final_cls_inds = get_detections(outputs[0][0], score_thresh, nms_thresh)
    t5 = time.time()

    # print(f"Preprocessing : {t2 - t1:.4f}s")
    # print(f"Inference     : {(t4 - t3) / 1:.4f}s")
    # print(f"Postprocessing: {t5 - t4:.4f}s")
    # print(f"Total         : {t2 - t1 + (t4 - t3) / 1 + t5 - t4:.4f}s")
    return final_boxes, final_scores, final_cls_inds

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

def detect_movement(boxes, prev_boxes, iou_threshold=0.3):
    movement_detected = False
    for box in boxes:
        for prev_box in prev_boxes:
            if calculate_iou(box, prev_box) < iou_threshold:
                movement_detected = True
                break
        if movement_detected:
            break
    return movement_detected

def process_latest_video():
    upload_folder = 'uploads'
    video_files = glob.glob(os.path.join(upload_folder, '*.mp4'))
    if not video_files:
        print("No video files found in upload folder.")
        return
    
    latest_video = max(video_files, key=os.path.getctime)
    print(f"Processing video: {latest_video}")

    score_thresh = 0.3
    nms_thresh = 0.45
    names_path = os.getenv('NAMES_PATH')
    model_path = os.getenv('MODEL_PATH')

    print(f"Model path: {model_path}")
    print(f"Names path: {names_path}")

    class_names = read_names(names_path)
    session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    cap = cv2.VideoCapture(latest_video)
    frame_index = 0
    save_dir = 'processed_frames'
    result_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    frame_size = None  # Initialize frame_size

    # Dictionary to count detections per class
    class_counts = {cls: 0 for cls in class_names}
    total_frames = 0
    car_movement_detected = False
    car_movement_frame = None
    prev_car_boxes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue  # Skip if frame is None

        final_boxes, final_scores, final_cls_ids = detect(session, frame)
        if final_boxes is not None:
            frame = vis(frame, final_boxes, final_scores, final_cls_ids, class_names=class_names)
            current_car_boxes = [box for box, cls_id in zip(final_boxes, final_cls_ids) if class_names[int(cls_id)] == 'car']
            if not car_movement_detected and detect_movement(current_car_boxes, prev_car_boxes):
                car_movement_detected = True
                car_movement_frame = frame_index / cap.get(cv2.CAP_PROP_FRAME_COUNT) * 100  # Calculate the percentage of video progress
            prev_car_boxes = current_car_boxes

            for cls_id in final_cls_ids:
                class_counts[class_names[int(cls_id)]] += 1
        
        if frame_size is None:  # Set frame_size
            frame_size = (frame.shape[1], frame.shape[0])
        
        image_path = os.path.join(save_dir, f'frame_{frame_index}.jpg')
        cv2.imwrite(image_path, frame)
        print(f"Saved processed frame to {image_path}")
        frame_index += 1
        total_frames += 1

    cap.release()

    if frame_size is None:
        print("No frames processed.")
        return

    # Calculate average detections per frame for each class
    avg_detections_per_frame = {cls: count / total_frames for cls, count in class_counts.items()}

    # Save results to a file
    results_file_path = os.path.join(result_dir, 'detection_results.txt')
    with open(results_file_path, 'w') as f:
        f.write("Average detections per frame:\n")
        for cls, avg_count in avg_detections_per_frame.items():
            f.write(f"{cls}: {avg_count:.2f}\n")
        f.write(f"\nCar movement detected: {'Yes' if car_movement_detected else 'No'}\n")
        if car_movement_detected:
            f.write(f"Car started moving at: {car_movement_frame:.2f} of the video progress\n")
    print(f"Results saved to {results_file_path}")

    # Create output video
    output_video_path = 'output/output_video.mp4'
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, frame_size)

    for i in range(frame_index):
        frame_path = os.path.join(save_dir, f'frame_{i}.jpg')
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()
    print(f"Processed video saved to {output_video_path}")

    # Create signal file to notify backend processing is complete
    signal_file_path = os.path.join('output', 'done.signal')
    with open(signal_file_path, 'w') as f:
        f.write('done')
    print(f"Signal file created at {signal_file_path}")

if __name__ == "__main__":
    process_latest_video()
