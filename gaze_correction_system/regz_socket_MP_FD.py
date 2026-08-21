# coding: utf-8
"""
FLX-Gaze: Real-time eye gaze correction system
Uses TensorFlow 2.x for neural network inference
Uses MediaPipe FaceLandmarker for face detection (cross-platform)
"""

import cv2
import sys
import os
import time
import socket
import struct
import numpy as np
import tensorflow as tf
from threading import Thread
import multiprocessing as mp
import pickle
import math

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from face_detection import FaceDetector

# Cross-platform screen detection
try:
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    Rs = (root.winfo_screenwidth(), root.winfo_screenheight())
    root.destroy()
except Exception:
    Rs = (1920, 1080)

# Enable TF1 compatibility mode
tf1 = tf.compat.v1
tf1.disable_eager_execution()

# Import model
import flx as model
from config import get_config
conf, _ = get_config()

# System parameters
model_dir = './' + conf.weight_set + '/warping_model/' + conf.mod + '/' + str(conf.ef_dim) + '/'
size_video = [640, 480]


class VideoReceiver:
    def __init__(self, shared_v, lock):
        self.close = False
        self.video_recv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')
        self.video_recv.bind(('', conf.recver_port))
        self.video_recv.listen(10)
        print('Socket now listening')
        self.conn, self.addr = self.video_recv.accept()

        # Face detection using MediaPipe
        self.detector = FaceDetector()
        self.start_recv(shared_v, lock)

    def face_detection(self, frame, shared_v, lock):
        """Detect face center for remote participant"""
        bbox = self.detector.detect_bbox(frame)
        coor_remote_head_center = [0, 0]

        if bbox:
            coor_remote_head_center = [
                (bbox[0] + bbox[2]) // 2,
                (bbox[1] + bbox[3]) // 2
            ]

        lock.acquire()
        shared_v[0] = coor_remote_head_center[0]
        shared_v[1] = coor_remote_head_center[1]
        lock.release()

    def start_recv(self, shared_v, lock):
        data = b""
        payload_size = struct.calcsize("L")
        print("payload_size: {}".format(payload_size))

        while True:
            while len(data) < payload_size:
                data += self.conn.recv(4096)

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += self.conn.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]
            frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")

            if frame == 'stop':
                print('stop')
                cv2.destroyWindow("Remote")
                break

            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            t = Thread(target=self.face_detection, args=(frame, shared_v, lock))
            t.start()

            cv2.imshow('Remote', frame)
            cv2.waitKey(1)

        self.detector.close()


class GazeRedirectionSystem:
    def __init__(self, shared_v, lock, auto_mode=False):
        # Face detection
        self.detector = FaceDetector()
        self.size_I = (48, 64)

        # Initial value
        self.Pe_z = -60

        # Get configurations
        self.f = conf.f
        self.Ps = (conf.S_W, conf.S_H)
        self.Pc = (conf.P_c_x, conf.P_c_y, conf.P_c_z)
        self.Pe = [self.Pc[0], self.Pc[1], self.Pe_z]

        # Start video sender
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client_socket.connect((conf.tar_ip, conf.sender_port))
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

        # Load model to GPU
        print("Loading model of [L] eye to GPU")
        self.L_sess, self.LE_input_img, self.LE_input_fp, \
            self.LE_input_ang, self.LE_phase_train, self.LE_img_pred = \
            self._load_model('L')

        print("Loading model of [R] eye to GPU")
        self.R_sess, self.RE_input_img, self.RE_input_fp, \
            self.RE_input_ang, self.RE_phase_train, self.RE_img_pred = \
            self._load_model('R')

        self.run(shared_v, lock, auto_mode)

    def _load_model(self, eye):
        with tf1.Graph().as_default() as g:
            with tf1.name_scope('inputs'):
                input_img = tf1.placeholder(
                    tf.float32, [None, conf.height, conf.width, conf.channel],
                    name="input_img"
                )
                input_fp = tf1.placeholder(
                    tf.float32, [None, conf.height, conf.width, conf.ef_dim],
                    name="input_fp"
                )
                input_ang = tf1.placeholder(
                    tf.float32, [None, conf.agl_dim], name="input_ang"
                )
                phase_train = tf1.placeholder(tf.bool, name='phase_train')

            img_pred, _, _ = model.inference(input_img, input_fp, input_ang,
                                              phase_train, conf)

            sess = tf1.Session(
                config=tf1.ConfigProto(allow_soft_placement=True),
                graph=g
            )

            saver = tf1.train.Saver(tf1.global_variables())
            ckpt = tf1.train.get_checkpoint_state(model_dir + eye + '/')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print(f'No checkpoint file found for {eye} eye')

            return (sess, input_img, input_fp, input_ang,
                    phase_train, img_pred)

    def monitor_para(self, frame, fig_alpha, fig_eye_pos, fig_R_w):
        cv2.rectangle(frame, (size_video[0] - 150, 0), (size_video[0], 55),
                      (255, 255, 255), -1)
        cv2.putText(frame,
                   'Eye:[' + str(int(fig_eye_pos[0])) + ',' +
                   str(int(fig_eye_pos[1])) + ',' + str(int(fig_eye_pos[2])) + ']',
                   (size_video[0] - 140, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame,
                   'alpha:[V=' + str(int(fig_alpha[0])) + ',H=' +
                   str(int(fig_alpha[1])) + ']',
                   (size_video[0] - 140, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame,
                   'R_w:[' + str(int(fig_R_w[0])) + ',' + str(int(fig_R_w[1])) + ']',
                   (size_video[0] - 140, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        return frame

    def get_inputs(self, frame, shape, pos="L", size_I=[48, 64]):
        """Get eye image and anchor maps using face landmarks"""
        from face_detection import RIGHT_EYE_INDICES, LEFT_EYE_INDICES

        if pos == "R":
            eye_indices = RIGHT_EYE_INDICES
            lc_idx, rc_idx = 33, 133
        elif pos == "L":
            eye_indices = LEFT_EYE_INDICES
            lc_idx, rc_idx = 362, 263
        else:
            print("Error: Wrong Eye")
            return None, None, None, None, None

        eye_cx = (shape.part(rc_idx).x + shape.part(lc_idx).x) * 0.5
        eye_cy = (shape.part(rc_idx).y + shape.part(lc_idx).y) * 0.5
        eye_center = [eye_cx, eye_cy]
        eye_len = abs(shape.part(rc_idx).x - shape.part(lc_idx).x)
        if eye_len < 1:
            eye_len = 1

        bx_d5w = eye_len * 3 / 4
        bx_h = 1.5 * bx_d5w
        sft_up = bx_h * 7 / 12
        sft_low = bx_h * 5 / 12

        y1 = max(0, int(eye_cy - sft_up))
        y2 = min(frame.shape[0], int(eye_cy + sft_low))
        x1 = max(0, int(eye_cx - bx_d5w))
        x2 = min(frame.shape[1], int(eye_cx + bx_d5w))

        img_eye = frame[y1:y2, x1:x2]
        if img_eye.size == 0:
            return None, None, None, None, None

        ori_size = [img_eye.shape[0], img_eye.shape[1]]
        LT_coor = [y1, x1]
        img_eye = cv2.resize(img_eye, (size_I[1], size_I[0]))

        # Create anchor maps
        ach_map = []
        for i, d in enumerate(eye_indices):
            resize_x = int((shape.part(d).x - LT_coor[1]) * size_I[1] / ori_size[1])
            resize_y = int((shape.part(d).y - LT_coor[0]) * size_I[0] / ori_size[0])

            ach_map_y = np.expand_dims(
                np.expand_dims(np.arange(0, size_I[0]) - resize_y, axis=1), axis=2)
            ach_map_y = np.tile(ach_map_y, [1, size_I[1], 1])

            ach_map_x = np.expand_dims(
                np.expand_dims(np.arange(0, size_I[1]) - resize_x, axis=0), axis=2)
            ach_map_x = np.tile(ach_map_x, [size_I[0], 1, 1])

            if i == 0:
                ach_map = np.concatenate((ach_map_x, ach_map_y), axis=2)
            else:
                ach_map = np.concatenate((ach_map, ach_map_x, ach_map_y), axis=2)

        return img_eye / 255, ach_map, eye_center, ori_size, LT_coor

    def shifting_angles_estimator(self, R_le, R_re, shared_v, lock):
        # Get P_w
        size_window = (659, 528)
        Rw_lt = [int(Rs[0] // 2 - size_window[0] // 2),
                 int(Rs[1] // 2 - size_window[1] // 2),
                 int(Rs[0] // 2 + size_window[0] // 2),
                 int(Rs[1] // 2 + size_window[1] // 2)]

        # Try to find Remote window (cross-platform)
        if sys.platform == 'win32':
            try:
                import win32gui
                tar_win = win32gui.FindWindow(None, "Remote")
                if tar_win:
                    Rw_lt = list(win32gui.GetWindowRect(tar_win))
                    size_window = (Rw_lt[2] - Rw_lt[0], Rw_lt[3] - Rw_lt[1])
            except Exception:
                pass

        pos_remote_head = [int(size_window[0] / 2), int(size_window[1] / 2)]

        try:
            if shared_v[0] != 0 and shared_v[1] != 0:
                pos_remote_head[0] = shared_v[0]
                pos_remote_head[1] = shared_v[1]
        except Exception:
            pass

        R_w = (Rw_lt[0] + pos_remote_head[0], Rw_lt[1] + pos_remote_head[1])
        Pw = (self.Ps[0] * (R_w[0] - Rs[0] / 2) / Rs[0],
              self.Ps[1] * (R_w[1] - Rs[1] / 2) / Rs[1], 0)

        # Get Pe
        eye_dist = np.sqrt((R_le[0] - R_re[0])**2 + (R_le[1] - R_re[1])**2)
        if eye_dist < 1:
            eye_dist = 1

        self.Pe[2] = -(self.f * conf.P_IDP) / eye_dist
        self.Pe[0] = (-np.abs(self.Pe[2]) *
                      (R_le[0] + R_re[0] - size_video[0]) /
                      (2 * self.f) + self.Pc[0])
        self.Pe[1] = (np.abs(self.Pe[2]) *
                      (R_le[1] + R_re[1] - size_video[1]) /
                      (2 * self.f) + self.Pc[1])

        a_w2z_x = math.degrees(math.atan(
            (Pw[0] - self.Pe[0]) / (Pw[2] - self.Pe[2])
        ))
        a_w2z_y = math.degrees(math.atan(
            (Pw[1] - self.Pe[1]) / (Pw[2] - self.Pe[2])
        ))
        a_z2c_x = math.degrees(math.atan(
            (self.Pe[0] - self.Pc[0]) / (self.Pc[2] - self.Pe[2])
        ))
        a_z2c_y = math.degrees(math.atan(
            (self.Pe[1] - self.Pc[1]) / (self.Pc[2] - self.Pe[2])
        ))

        alpha = [int(a_w2z_y + a_z2c_y), int(a_w2z_x + a_z2c_x)]
        return alpha, self.Pe, R_w

    def flx_gaze(self, frame, shape, shared_v, lock,
                 pixel_cut=[3, 4], size_I=[48, 64]):
        alpha_w2c = [0, 0]
        R_w = [0, 0]

        if shape is None:
            return False

        # Get eye regions
        LE_img, LE_M_A, LE_center, size_le_ori, R_le_LT = \
            self.get_inputs(frame, shape, pos="L", size_I=size_I)
        RE_img, RE_M_A, RE_center, size_re_ori, R_re_LT = \
            self.get_inputs(frame, shape, pos="R", size_I=size_I)

        if LE_img is None or RE_img is None:
            return False

        # Shifting angles estimator
        alpha_w2c, _, R_w = self.shifting_angles_estimator(
            LE_center, RE_center, shared_v, lock
        )

        # Left Eye inference
        LE_infer_img = self.L_sess.run(
            self.LE_img_pred,
            feed_dict={
                self.LE_input_img: np.expand_dims(LE_img, axis=0),
                self.LE_input_fp: np.expand_dims(LE_M_A, axis=0),
                self.LE_input_ang: np.expand_dims(alpha_w2c, axis=0),
                self.LE_phase_train: False
            }
        )
        LE_infer = cv2.resize(
            LE_infer_img.reshape(size_I[0], size_I[1], 3),
            (size_le_ori[1], size_le_ori[0])
        )

        # Right Eye inference
        RE_infer_img = self.R_sess.run(
            self.RE_img_pred,
            feed_dict={
                self.RE_input_img: np.expand_dims(RE_img, axis=0),
                self.RE_input_fp: np.expand_dims(RE_M_A, axis=0),
                self.RE_input_ang: np.expand_dims(alpha_w2c, axis=0),
                self.RE_phase_train: False
            }
        )
        RE_infer = cv2.resize(
            RE_infer_img.reshape(size_I[0], size_I[1], 3),
            (size_re_ori[1], size_re_ori[0])
        )

        # Replace eyes in frame
        frame[
            (R_le_LT[0] + pixel_cut[0]):(R_le_LT[0] + size_le_ori[0] - pixel_cut[0]),
            (R_le_LT[1] + pixel_cut[1]):(R_le_LT[1] + size_le_ori[1] - pixel_cut[1])
        ] = LE_infer[pixel_cut[0]:(-1 * pixel_cut[0]),
                     pixel_cut[1]:(-1 * pixel_cut[1])] * 255

        frame[
            (R_re_LT[0] + pixel_cut[0]):(R_re_LT[0] + size_re_ori[0] - pixel_cut[0]),
            (R_re_LT[1] + pixel_cut[1]):(R_re_LT[1] + size_re_ori[1] - pixel_cut[1])
        ] = RE_infer[pixel_cut[0]:(-1 * pixel_cut[0]),
                     pixel_cut[1]:(-1 * pixel_cut[1])] * 255

        frame = self.monitor_para(frame, alpha_w2c, self.Pe, R_w)

        result, imgencode = cv2.imencode('.jpg', frame, self.encode_param)
        data = pickle.dumps(imgencode, 0)
        self.client_socket.sendall(struct.pack("L", len(data)) + data)
        return True

    def redirect_gaze(self, frame, shared_v, lock):
        """Detect face and redirect gaze"""
        shape = self.detector.detect(frame)
        if shape is None:
            return False

        rg_thread = Thread(target=self.flx_gaze,
                          args=(frame, shape, shared_v, lock))
        rg_thread.start()
        return True

    def run(self, shared_v, lock, auto_mode=False):
        redir = auto_mode
        size_window = [659, 528]
        vs = cv2.VideoCapture(0)
        vs.set(3, size_video[0])
        vs.set(4, size_video[1])
        t = time.time()

        cv2.namedWindow(conf.uid)
        cv2.moveWindow(conf.uid,
                      int(Rs[0] / 2) - int(size_window[0] / 2),
                      int(Rs[1] / 2) - int(size_window[1] / 2))

        while True:
            ret, recv_frame = vs.read()
            if ret:
                cv2.imshow(conf.uid, recv_frame)
                if recv_frame is not None:
                    if redir:
                        frame = recv_frame.copy()
                        try:
                            self.redirect_gaze(frame, shared_v, lock)
                        except Exception as e:
                            print(f"Error: {e}")
                    else:
                        result, imgencode = cv2.imencode(
                            '.jpg', recv_frame, self.encode_param
                        )
                        data = pickle.dumps(imgencode, 0)
                        self.client_socket.sendall(
                            struct.pack("L", len(data)) + data
                        )

                if (time.time() - t) > 1:
                    t = time.time()

                k = cv2.waitKey(10)
                if k == ord('q'):
                    data = pickle.dumps('stop')
                    self.client_socket.sendall(
                        struct.pack("L", len(data)) + data
                    )
                    time.sleep(3)
                    cv2.destroyWindow(conf.uid)
                    self.client_socket.shutdown(socket.SHUT_RDWR)
                    self.client_socket.close()
                    vs.release()
                    self.L_sess.close()
                    self.R_sess.close()
                    self.detector.close()
                    break
                elif k == ord('r'):
                    redir = not redir


if __name__ == '__main__':
    auto_mode = '--auto' in sys.argv

    print("=" * 60)
    print("FLX-Gaze: Eye Gaze Correction System")
    print("=" * 60)
    if auto_mode:
        print("Auto mode: Gaze correction ON by default")
    print("Press 'r' to toggle, 'q' to quit")
    print("=" * 60)

    l = mp.Lock()
    v = mp.Array('i', [320, 240])

    vs_thread = mp.Process(target=VideoReceiver, args=(v, l))
    vs_thread.start()
    time.sleep(1)

    gz_thread = mp.Process(target=GazeRedirectionSystem, args=(v, l, auto_mode))
    gz_thread.start()

    vs_thread.join()
    gz_thread.join()
