"""Full integration test for FLX-Gaze system with TF2."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import traceback

# Track results
results = []

def test(name, fn):
    try:
        fn()
        results.append((name, "PASS", ""))
        print(f"  [PASS] {name}")
    except Exception as e:
        results.append((name, "FAIL", str(e)))
        print(f"  [FAIL] {name}: {e}")

# ============================================================
print("=" * 60)
print("TEST 1: Import all modules")
print("=" * 60)
# ============================================================

import tensorflow as tf
tf1 = tf.compat.v1
tf1.disable_eager_execution()

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gaze_correction_system"))

import config
import tf_utils
import transformation
import flx

def t_imports():
    assert tf.__version__.startswith("2."), f"TF version: {tf.__version__}"
    assert np.__version__, "NumPy missing"
    assert cv2.__version__, "OpenCV missing"
test("all imports + TF2 version", t_imports)

# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Config parsing")
print("=" * 60)
# ============================================================

def t_config():
    conf, _ = config.get_config()
    assert conf.height == 48
    assert conf.width == 64
    assert conf.channel == 3
    assert conf.ef_dim == 12
    assert conf.agl_dim == 2
    assert conf.encoded_agl_dim == 16
    assert conf.f == 650
    assert conf.P_IDP == 6.3
test("config parsing", t_config)

# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Model graph construction")
print("=" * 60)
# ============================================================

def t_graph():
    tf1.reset_default_graph()
    conf, _ = config.get_config()
    input_img = tf1.placeholder(tf.float32, [None, 48, 64, 3])
    input_fp = tf1.placeholder(tf.float32, [None, 48, 64, 12])
    input_agl = tf1.placeholder(tf.float32, [None, 2])
    phase_train = tf1.placeholder(tf.bool)

    img_pred, flow_raw, lcm_map = flx.inference(
        input_img, input_fp, input_agl, phase_train, conf
    )
    assert img_pred.shape.as_list() == [None, 48, 64, 3]
    assert flow_raw.shape.as_list() == [None, 48, 64, 2]
    assert lcm_map.shape.as_list() == [None, 48, 64, 2]
test("graph construction", t_graph)

# ============================================================
print("\n" + "=" * 60)
print("TEST 4: Forward pass with dummy data")
print("=" * 60)
# ============================================================

def t_forward():
    tf1.reset_default_graph()
    conf, _ = config.get_config()
    input_img = tf1.placeholder(tf.float32, [None, 48, 64, 3])
    input_fp = tf1.placeholder(tf.float32, [None, 48, 64, 12])
    input_agl = tf1.placeholder(tf.float32, [None, 2])
    phase_train = tf1.placeholder(tf.bool)

    img_pred, flow_raw, lcm_map = flx.inference(
        input_img, input_fp, input_agl, phase_train, conf
    )

    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())
        pred, flow, lcm = sess.run(
            [img_pred, flow_raw, lcm_map],
            feed_dict={
                input_img: np.random.randn(1, 48, 64, 3).astype(np.float32),
                input_fp: np.random.randn(1, 48, 64, 12).astype(np.float32),
                input_agl: np.array([[10, 5]], dtype=np.float32),
                phase_train: False
            }
        )
        assert pred.shape == (1, 48, 64, 3)
        assert flow.shape == (1, 48, 64, 2)
        assert lcm.shape == (1, 48, 64, 2)
        assert np.isfinite(pred).all(), "pred has NaN/Inf"
        lcm_sum = lcm.sum(axis=3)
        assert np.allclose(lcm_sum, 1.0, atol=1e-4), f"LCM sum: {lcm_sum.mean():.6f}"
test("forward pass", t_forward)

# ============================================================
print("\n" + "=" * 60)
print("TEST 5: Batch forward pass (batch=4)")
print("=" * 60)
# ============================================================

def t_batch():
    tf1.reset_default_graph()
    conf, _ = config.get_config()
    input_img = tf1.placeholder(tf.float32, [None, 48, 64, 3])
    input_fp = tf1.placeholder(tf.float32, [None, 48, 64, 12])
    input_agl = tf1.placeholder(tf.float32, [None, 2])
    phase_train = tf1.placeholder(tf.bool)

    img_pred, flow_raw, lcm_map = flx.inference(
        input_img, input_fp, input_agl, phase_train, conf
    )

    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())
        pred, flow, lcm = sess.run(
            [img_pred, flow_raw, lcm_map],
            feed_dict={
                input_img: np.random.randn(4, 48, 64, 3).astype(np.float32),
                input_fp: np.random.randn(4, 48, 64, 12).astype(np.float32),
                input_agl: np.array([[10, 5], [-5, 10], [0, 0], [15, -15]], dtype=np.float32),
                phase_train: False
            }
        )
        assert pred.shape == (4, 48, 64, 3)
        assert flow.shape == (4, 48, 64, 2)
        assert lcm.shape == (4, 48, 64, 2)
test("batch forward (batch=4)", t_batch)

# ============================================================
print("\n" + "=" * 60)
print("TEST 6: Training vs eval mode")
print("=" * 60)
# ============================================================

def t_train_eval():
    tf1.reset_default_graph()
    conf, _ = config.get_config()
    input_img = tf1.placeholder(tf.float32, [None, 48, 64, 3])
    input_fp = tf1.placeholder(tf.float32, [None, 48, 64, 12])
    input_agl = tf1.placeholder(tf.float32, [None, 2])
    phase_train = tf1.placeholder(tf.bool)

    img_pred, flow_raw, lstm_map = flx.inference(
        input_img, input_fp, input_agl, phase_train, conf
    )

    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())
        d = {
            input_img: np.random.randn(2, 48, 64, 3).astype(np.float32),
            input_fp: np.random.randn(2, 48, 64, 12).astype(np.float32),
            input_agl: np.array([[10, 5], [-5, 10]], dtype=np.float32),
        }

        pred_train = sess.run(img_pred, feed_dict={**d, phase_train: True})
        pred_eval = sess.run(img_pred, feed_dict={**d, phase_train: False})

        assert pred_train.shape == (2, 48, 64, 3)
        assert pred_eval.shape == (2, 48, 64, 3)
test("training vs eval mode", t_train_eval)

# ============================================================
print("\n" + "=" * 60)
print("TEST 7: Geometric model")
print("=" * 60)
# ============================================================

import math

def t_geometric():
    R_le = [280, 240]
    R_re = [360, 240]
    f = 650
    P_IDP = 6.3
    Pc = (0, -21, -1)
    eye_dist = math.sqrt((R_le[0]-R_re[0])**2 + (R_le[1]-R_re[1])**2)
    assert eye_dist == 80.0
    Pe_z = -(f * P_IDP) / eye_dist
    Pe_x = -abs(Pe_z) * (R_le[0]+R_re[0]-640) / (2*f) + Pc[0]
    Pe_y = abs(Pe_z) * (R_le[1]+R_re[1]-480) / (2*f) + Pc[1]
    assert Pe_z < 0
    assert abs(Pe_x) < 10
test("geometric model", t_geometric)

# ============================================================
print("\n" + "=" * 60)
print("TEST 8: Anchor maps")
print("=" * 60)
# ============================================================

def t_anchor():
    size_I = [48, 64]
    landmarks = [(280, 230), (295, 220), (310, 220),
                 (325, 230), (310, 240), (295, 240)]
    FP_seq = [36, 37, 38, 39, 40, 41]
    eye_cx = np.mean([l[0] for l in landmarks])
    eye_cy = np.mean([l[1] for l in landmarks])
    eye_len = abs(landmarks[3][0] - landmarks[0][0])
    bx_d5w = eye_len * 3/4
    bx_h = 1.5 * bx_d5w
    sft_up = bx_h * 7/12
    ori_size = [int(bx_h), int(bx_d5w * 2)]
    LT_coor = [int(eye_cy - sft_up), int(eye_cx - bx_d5w)]
    ach_map = []
    for i, d in enumerate(FP_seq):
        resize_x = int((landmarks[i][0] - LT_coor[1]) * size_I[1] / ori_size[1])
        resize_y = int((landmarks[i][1] - LT_coor[0]) * size_I[0] / ori_size[0])
        ach_map_y = np.tile(np.expand_dims(np.expand_dims(np.arange(0, size_I[0]) - resize_y, axis=1), axis=2), [1, size_I[1], 1])
        ach_map_x = np.tile(np.expand_dims(np.expand_dims(np.arange(0, size_I[1]) - resize_x, axis=0), axis=2), [size_I[0], 1, 1])
        ach_map = np.concatenate((ach_map_x, ach_map_y), axis=2) if i == 0 else np.concatenate((ach_map, ach_map_x, ach_map_y), axis=2)
    assert ach_map.shape == (48, 64, 12)
test("anchor map generation", t_anchor)

# ============================================================
print("\n" + "=" * 60)
print("TEST 9: Loss functions")
print("=" * 60)
# ============================================================

def t_loss():
    tf1.reset_default_graph()
    pred = tf1.placeholder(tf.float32, [None, 48, 64, 3])
    gt = tf1.placeholder(tf.float32, [None, 48, 64, 3])
    l2 = flx.dist_loss(pred, gt, method="L2")
    mae = flx.dist_loss(pred, gt, method="MAE")

    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())
        d = {pred: np.random.randn(2, 48, 64, 3).astype(np.float32),
             gt: np.random.randn(2, 48, 64, 3).astype(np.float32)}
        l2_val, mae_val = sess.run([l2, mae], feed_dict=d)
        assert l2_val >= 0
        assert mae_val >= 0
test("loss functions", t_loss)

# ============================================================
print("\n" + "=" * 60)
print("TEST 10: Edge cases")
print("=" * 60)
# ============================================================

def t_edge():
    tf1.reset_default_graph()
    conf, _ = config.get_config()
    input_img = tf1.placeholder(tf.float32, [None, 48, 64, 3])
    input_fp = tf1.placeholder(tf.float32, [None, 48, 64, 12])
    input_agl = tf1.placeholder(tf.float32, [None, 2])
    phase_train = tf1.placeholder(tf.bool)

    img_pred, flow_raw, lcm_map = flx.inference(
        input_img, input_fp, input_agl, phase_train, conf
    )

    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())
        zero = np.zeros((1, 48, 64, 3), dtype=np.float32)
        zero_fp = np.zeros((1, 48, 64, 12), dtype=np.float32)

        for angle in [[0, 0], [40, 30], [-40, -30]]:
            p = sess.run(img_pred, feed_dict={
                input_img: zero, input_fp: zero_fp,
                input_agl: np.array([angle], dtype=np.float32),
                phase_train: False
            })
            assert p.shape == (1, 48, 64, 3)
test("edge cases (various angles)", t_edge)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
total = len(results)

for name, status, err in results:
    tag = "OK" if status == "PASS" else "FAIL"
    extra = f" -- {err}" if err else ""
    print(f"  [{tag}] {name}{extra}")

print(f"\n  {passed}/{total} passed, {failed} failed")

if failed == 0:
    print("\n  ALL TESTS PASSED!")
else:
    print(f"\n  {failed} TEST(S) FAILED!")
    sys.exit(1)
