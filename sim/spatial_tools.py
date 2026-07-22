"""
Spatial tools for LIBERO harness — image-based LocateBox + Jacobian MoveTo.
Mirrors the FR3 spatial path (cortex-droid-v2.ts). No privileged sim state used at runtime.

Usage (within harness):
    from spatial_tools import unproject_bbox_to_world, move_ee_to

    # LocateBox: bbox center → world coords via depth + camera intrinsics
    world_xyz = unproject_bbox_to_world(env, bbox=(x1,y1,x2,y2), camera="agentview")

    # MoveTo: drive EE to target position via Jacobian iteration
    success = move_ee_to(env, target_pos=world_xyz + np.array([0,0,0.10]), max_steps=200)
"""
import numpy as np


# ===== SAM (Segment Anything) integration for robust depth sampling =====
# Lazy-loaded module-level cache for the SAM model + processor.
# Given a bbox, SAM returns a per-pixel mask of the object inside it; we then
# sample depth ONLY on mask pixels, excluding the floor / background that the
# bbox unavoidably includes for tall objects. Solves the classic "bbox depth
# falls to floor because bbox includes both object and table" failure.
_SAM_CACHE = {"model": None, "processor": None}


def _sam_get_mask(rgb_image, bbox, llm_view_rotated=True, img_h=256, img_w=256):
    """Run SAM with a bbox prompt. Returns a (H, W) bool mask (True = object).
    `rgb_image` should be the agentview image in NATIVE orientation (i.e., NOT
    rotated 180°). `bbox` is in LLM-view coords; we rotate it back to native
    before passing to SAM if llm_view_rotated=True. Returns None on any failure
    (caller should fall back to non-SAM depth sampling)."""
    try:
        if _SAM_CACHE["model"] is None:
            from transformers import SamModel, SamProcessor
            import torch
            print("[sam] loading facebook/sam-vit-base ...", flush=True)
            _SAM_CACHE["processor"] = SamProcessor.from_pretrained("facebook/sam-vit-base")
            _SAM_CACHE["model"] = SamModel.from_pretrained("facebook/sam-vit-base").to("cuda").eval()
            print("[sam] loaded", flush=True)
        import torch
        from PIL import Image as _Image
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if llm_view_rotated:
            x1n, x2n = (img_w - 1) - x2, (img_w - 1) - x1
            y1n, y2n = y1, y2
        else:
            x1n, x2n, y1n, y2n = x1, x2, y1, y2
        x1n = max(0, min(img_w - 1, x1n)); x2n = max(0, min(img_w - 1, x2n))
        y1n = max(0, min(img_h - 1, y1n)); y2n = max(0, min(img_h - 1, y2n))
        if x1n > x2n: x1n, x2n = x2n, x1n
        if y1n > y2n: y1n, y2n = y2n, y1n
        pil = _Image.fromarray(rgb_image)
        input_boxes = [[[float(x1n), float(y1n), float(x2n), float(y2n)]]]  # SAM input format
        inputs = _SAM_CACHE["processor"](
            pil, input_boxes=input_boxes, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            outputs = _SAM_CACHE["model"](**inputs, multimask_output=False)
        masks = _SAM_CACHE["processor"].image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        # masks: list[Tensor(num_boxes, num_masks, H, W)] — pull first
        m = masks[0][0, 0].cpu().numpy().astype(bool)
        return m
    except Exception as e:
        print(f"[sam] mask failed: {e}", flush=True)
        return None


def get_real_depth(env, camera_name: str, depth_map: np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo's normalized z-buffer (depth_map values in [0, 1]) to true
    metric depth using the sim's near/far/extent parameters.
    """
    extent = env.sim.model.stat.extent
    near = env.sim.model.vis.map.znear * extent
    far = env.sim.model.vis.map.zfar * extent
    # depth in [0,1]; closer to 1 = far. Convert via OpenGL-style formula:
    return near / (1.0 - depth_map * (1.0 - near / far))


def _raycast_unproject(env, px_native, py_native, img_h=512, img_w=512, camera="agentview"):
    """Ray-cast from agentview camera through native pixel (px, py); return world hit point."""
    import mujoco
    import robosuite.utils.camera_utils as _CU
    sim = env.sim
    K = _CU.get_camera_intrinsic_matrix(sim, camera, img_h, img_w)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    cam_id = sim.model.camera_name2id(camera)
    cam_pos = np.array(sim.data.cam_xpos[cam_id], dtype=float)
    cam_mat = np.array(sim.data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
    x_norm = (px_native - cx) / fx
    y_norm = (py_native - cy) / fy
    ray_cam = np.array([x_norm, -y_norm, -1.0])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)
    ray_world = cam_mat @ ray_cam
    m = sim.model._model
    d = sim.data._data
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(m, d, cam_pos, ray_world, None, 1, -1, geomid)
    if dist < 0:
        return None
    return cam_pos + dist * ray_world


def unproject_bbox_to_world(env, bbox, camera="agentview", img_h=256, img_w=256,
                            depth_arr=None, llm_view_rotated=True, rgb_arr=None,
                            use_sam=True):
    """
    bbox = (x1, y1, x2, y2) in pixels.

    If `llm_view_rotated` is True, the bbox coords are in the image AFTER the
    [::-1, ::-1] 180° rotation that OpenPI / our harness applies before
    rendering for the LLM/policy. We undo that rotation to get pixel coords in
    the native depth-image frame.

    If `use_sam` is True (default), we use SAM (Segment Anything) to get a
    per-pixel mask of the object inside the bbox, then sample depth ONLY on
    mask pixels. This solves the failure mode where the bbox includes both
    object and table/floor pixels, and median/percentile depth picks the floor.
    Requires the agentview RGB image; if `rgb_arr` is None we render it.
    """
    import robosuite.utils.camera_utils as CU

    x1, y1, x2, y2 = [int(v) for v in bbox]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    if depth_arr is None:
        depth_arr = env.sim.render(camera_name=camera, height=img_h, width=img_w, depth=True)[1]
    depth_norm = np.asarray(depth_arr).squeeze()
    if depth_norm.shape[0] != img_h or depth_norm.shape[1] != img_w:
        raise ValueError(f"depth shape {depth_norm.shape} != ({img_h},{img_w})")

    # Map LLM-view bbox corners back to native depth-image coords
    if llm_view_rotated:
        # FIX: world_to_pix returns col that's H-mirrored from LLM view, row is unchanged.
        # The original "180-rotation un-flip" was over-correcting in Y.
        x1n, x2n = (img_w - 1) - x2, (img_w - 1) - x1
        y1n, y2n = y1, y2
    else:
        x1n, x2n = x1, x2
        y1n, y2n = y1, y2
    x1n = max(0, min(img_w - 1, x1n)); x2n = max(0, min(img_w - 1, x2n))
    y1n = max(0, min(img_h - 1, y1n)); y2n = max(0, min(img_h - 1, y2n))
    if x1n > x2n: x1n, x2n = x2n, x1n
    if y1n > y2n: y1n, y2n = y2n, y1n

    real_depth = CU.get_real_depth_map(env.sim, depth_norm)  # (H, W)

    # ===== SAM-based mask depth sampling (preferred path) =====
    sam_mask = None
    if use_sam:
        if rgb_arr is None:
            try:
                rgb_arr = env.sim.render(camera_name=camera, height=img_h, width=img_w)
                # robosuite renders flipped; we want native orientation matching depth
                rgb_arr = np.asarray(rgb_arr)
            except Exception:
                rgb_arr = None
        if rgb_arr is not None:
            sam_mask = _sam_get_mask(rgb_arr, bbox, llm_view_rotated=llm_view_rotated,
                                      img_h=img_h, img_w=img_w)

    # RAYCAST PATH: when we have a SAM mask, ray-cast through the LARGEST
    # connected component's centroid (avoids clutter where mask spans multiple
    # objects and centroid drifts to a neighbor).
    if sam_mask is not None and sam_mask[y1n:y2n+1, x1n:x2n+1].sum() > 5:
        bbox_mask_check = sam_mask[y1n:y2n+1, x1n:x2n+1].astype(bool)
        try:
            from scipy.ndimage import label as _cc_label
            _labels, _n = _cc_label(bbox_mask_check)
            if _n == 0:
                _mys, _mxs = np.where(bbox_mask_check)
            elif _n == 1:
                _mys, _mxs = np.where(bbox_mask_check)
            else:
                # Pick the largest component
                _sizes = np.bincount(_labels.ravel())
                _sizes[0] = 0  # background
                _largest = int(_sizes.argmax())
                _mys, _mxs = np.where(_labels == _largest)
        except Exception:
            _mys, _mxs = np.where(bbox_mask_check)
        if len(_mys) > 0:
            _cpx = x1n + float(np.mean(_mxs))
            _cpy = y1n + float(np.mean(_mys))
            _rc = _raycast_unproject(env, _cpx, _cpy, img_h=img_h, img_w=img_w, camera=camera)
            if _rc is not None:
                return _rc

    if sam_mask is not None and sam_mask[y1n:y2n+1, x1n:x2n+1].sum() > 5:
        # SAM gave us a usable mask. Use:
        #   (x, y) = CENTROID of the mask (tight object center, not bbox center)
        #   z     = depth at the TOP of the mask (smallest depth, = object top surface)
        # Top-of-object depth + mask centroid gives accurate top-down grasp targets
        # for thin objects where the bbox is much larger than the object footprint.
        bbox_depth = real_depth[y1n:y2n + 1, x1n:x2n + 1]
        bbox_mask = sam_mask[y1n:y2n + 1, x1n:x2n + 1]
        mask_ys, mask_xs = np.where(bbox_mask)
        # Mask centroid in BBOX-local coords:
        cy_local = int(round(float(np.mean(mask_ys))))
        cx_local = int(round(float(np.mean(mask_xs))))
        # Depth: use the 10th percentile of mask depths (biased toward the
        # CLOSEST-to-camera = TOP surface of the object). This works for any
        # shape: bowl rim (top of concave bowl), bottle top, box top, flat plate
        # — all are captured by the closest 10% of mask pixels. The previous
        # 25th-percentile was biased toward the bowl interior because the SAM
        # mask sees pixels through the bowl opening down to the table.
        # Robust against single-pixel min noise.
        masked_depths = bbox_depth[bbox_mask]
        target_d = float(np.percentile(masked_depths, 10))
        argmin_y = y1n + cy_local
        argmin_x = x1n + cx_local
        # Override depth at centroid with the object-top depth we want to grasp
        # against. We do this by directly modifying real_depth at the centroid
        # pixel — robosuite's unproject reads real_depth[argmin_y, argmin_x].
        # (We DON'T want to modify a global tensor in-place; instead we'll
        # construct a tiny patched depth map below.)
        # Actually, simplest: just substitute the target depth in a local copy.
        real_depth = real_depth.copy()
        real_depth[argmin_y, argmin_x] = target_d
    else:
        # Fallback: low-percentile center-window sampling (no SAM available)
        window = real_depth[y1n:y2n + 1, x1n:x2n + 1]
        if window.size == 0:
            cy_native = (y1n + y2n) // 2
            cx_native = (x1n + x2n) // 2
            argmin_y, argmin_x = cy_native, cx_native
        else:
            cy_w = (y1n + y2n) // 2
            cx_w = (x1n + x2n) // 2
            r_y = max(1, min(4, (y2n - y1n) // 5))
            r_x = max(1, min(4, (x2n - x1n) // 5))
            ys1, ys2 = max(0, cy_w - r_y), min(img_h - 1, cy_w + r_y)
            xs1, xs2 = max(0, cx_w - r_x), min(img_w - 1, cx_w + r_x)
            sub = real_depth[ys1:ys2 + 1, xs1:xs2 + 1]
            target_d = float(np.percentile(sub, 10))
            diff = np.abs(sub - target_d)
            idx = int(np.argmin(diff))
            ny, nx = np.unravel_index(idx, sub.shape)
            argmin_y = ys1 + int(ny)
            argmin_x = xs1 + int(nx)

    # robosuite's get_camera_transform_matrix returns the WORLD-TO-PIXEL 4x4
    # transform. transform_from_pixels_to_world wants the inverse.
    world_to_pix = CU.get_camera_transform_matrix(env.sim, camera, img_h, img_w)
    pix_to_world = np.linalg.inv(world_to_pix)

    pixels = np.array([argmin_y, argmin_x], dtype=float)
    world_xyz = CU.transform_from_pixels_to_world(
        pixels=pixels,
        depth_map=real_depth[..., None],
        camera_to_world_transform=pix_to_world,
    )
    return np.asarray(world_xyz, dtype=np.float64)


def get_object_point_cloud_and_pose(env, bbox, camera="agentview", img_h=256, img_w=256,
                                      depth_arr=None, rgb_arr=None, llm_view_rotated=True):
    """Generate an oriented grasp pose from a SAM-masked point cloud (PCA-based).

    Returns dict:
      centroid: (x, y, z) world centroid of object
      top_z: z of object top (for descend target)
      yaw_rad: gripper yaw (rotation about world z) that aligns gripper's grasp axis
               with the object's SHORT principal axis in xy plane. radians.
      width: estimated grasp width (along the short axis, meters)
      points: Nx3 numpy array of all mask-pixel world points (for debugging)
    """
    import robosuite.utils.camera_utils as CU

    x1, y1, x2, y2 = [int(v) for v in bbox]
    if depth_arr is None:
        depth_arr = env.sim.render(camera_name=camera, height=img_h, width=img_w, depth=True)[1]
    depth_norm = np.asarray(depth_arr).squeeze()
    real_depth = CU.get_real_depth_map(env.sim, depth_norm)

    # Rotate bbox to native frame.
    if llm_view_rotated:
        # FIX: world_to_pix returns col that's H-mirrored from LLM view, row is unchanged.
        # The original "180-rotation un-flip" was over-correcting in Y.
        x1n, x2n = (img_w - 1) - x2, (img_w - 1) - x1
        y1n, y2n = y1, y2
    else:
        x1n, x2n = x1, x2
        y1n, y2n = y1, y2
    x1n = max(0, min(img_w - 1, x1n)); x2n = max(0, min(img_w - 1, x2n))
    y1n = max(0, min(img_h - 1, y1n)); y2n = max(0, min(img_h - 1, y2n))
    if x1n > x2n: x1n, x2n = x2n, x1n
    if y1n > y2n: y1n, y2n = y2n, y1n

    # Get SAM mask
    if rgb_arr is None:
        rgb_arr = env.sim.render(camera_name=camera, height=img_h, width=img_w)
    sam_mask = _sam_get_mask(rgb_arr, bbox, llm_view_rotated=llm_view_rotated,
                              img_h=img_h, img_w=img_w)
    if sam_mask is None or sam_mask[y1n:y2n+1, x1n:x2n+1].sum() < 10:
        return None

    # Build point cloud from mask pixels.
    world_to_pix = CU.get_camera_transform_matrix(env.sim, camera, img_h, img_w)
    pix_to_world = np.linalg.inv(world_to_pix)

    mask_ys, mask_xs = np.where(sam_mask)
    # Filter to inside the bbox to avoid SAM bleed
    inside = (mask_ys >= y1n) & (mask_ys <= y2n) & (mask_xs >= x1n) & (mask_xs <= x2n)
    mask_ys = mask_ys[inside]; mask_xs = mask_xs[inside]
    if len(mask_ys) < 10:
        return None

    # subsample for speed if huge
    if len(mask_ys) > 800:
        idx = np.random.choice(len(mask_ys), 800, replace=False)
        mask_ys = mask_ys[idx]; mask_xs = mask_xs[idx]

    # Unproject each masked pixel to world
    pixels = np.stack([mask_ys, mask_xs], axis=1).astype(float)
    # Inline unprojection (robosuite's transform_from_pixels_to_world has
    # an assert that breaks for (N, 2) pixels + single (H, W, 1) depth).
    _ys_i = pixels[:, 0].astype(int).clip(0, real_depth.shape[0] - 1)
    _xs_i = pixels[:, 1].astype(int).clip(0, real_depth.shape[1] - 1)
    _zs = real_depth[_ys_i, _xs_i].astype(float)
    _cam_pts = np.stack([_xs_i * _zs, _ys_i * _zs, _zs, np.ones_like(_zs)], axis=1)
    world_pts = (pix_to_world @ _cam_pts.T).T[:, :3]
    world_pts = np.asarray(world_pts, dtype=np.float64)  # (N, 3)

    # Reject obvious outliers (points below floor or above ceiling)
    keep = (world_pts[:, 2] > -0.05) & (world_pts[:, 2] < 0.6)
    if keep.sum() < 10:
        return None
    world_pts = world_pts[keep]

    centroid = world_pts.mean(axis=0)
    # PCA on xy points only (gripper rotates about world z axis)
    xy = world_pts[:, :2] - centroid[:2]
    cov = np.cov(xy.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # sorted ascending
    # Short axis = first eigvec (smallest eigval). Grasp axis aligns with short axis.
    short_axis = eigvecs[:, 0]
    long_axis = eigvecs[:, 1]
    yaw_rad = float(np.arctan2(short_axis[1], short_axis[0]))
    # Grasp width: 2× sqrt(largest eigval along short axis) — half-extent doubled
    # Eigvals are variance; sqrt is std-dev. ~2-sigma covers most of the object.
    width = float(2.5 * np.sqrt(max(eigvals[0], 1e-6)))
    width = min(max(width, 0.02), 0.08)  # clamp to gripper range
    top_z = float(np.percentile(world_pts[:, 2], 90))  # near-top (90th pct)

    return {
        "centroid": (float(centroid[0]), float(centroid[1]), float(centroid[2])),
        "top_z": top_z,
        "yaw_rad": yaw_rad,
        "width": width,
        "points_n": int(len(world_pts)),
        "long_extent": float(2.5 * np.sqrt(max(eigvals[1], 1e-6))),
    }


def project_world_to_pixel(env, world_xyz, camera="agentview", img_h=256, img_w=256,
                           llm_view_rotated=True):
    """For sanity-test: project a world point into pixel coords using robosuite's helper."""
    import robosuite.utils.camera_utils as CU
    pts = np.array([world_xyz])
    px = CU.project_points_from_world_to_camera(
        points=pts,
        world_to_camera_transform=CU.get_camera_transform_matrix(env.sim, camera, img_h, img_w),
        camera_height=img_h,
        camera_width=img_w,
    )[0]
    # px is (row, col)
    row, col = int(px[0]), int(px[1])
    if llm_view_rotated:
        col = (img_w - 1) - col
    return col, row  # return (x, y) = (col, row)


def get_current_ee_pos(env) -> np.ndarray:
    """Read robot0 EE position from sim."""
    body_id = env.sim.model.body_name2id("gripper0_eef")
    return np.array(env.sim.data.body_xpos[body_id])


def grasp(env, n_steps=20):
    """Close gripper for n_steps — no VLA. Holds EE in place (zero pos/rot deltas)."""
    for _ in range(n_steps):
        env.step([0.0]*6 + [1.0])  # gripper close


def release(env, n_steps=20):
    """Open gripper for n_steps — no VLA. Holds EE in place."""
    for _ in range(n_steps):
        env.step([0.0]*6 + [-1.0])  # gripper open


def move_ee_to(env, target_pos, max_steps=200, pos_tol=0.02, gripper_open=True,
               osc_pos_scale=0.05):
    """
    Drive the EE toward target_pos using OSC_POSE Cartesian deltas (the env's controller).
    Action layout: [dx, dy, dz, drx, dry, drz, grip], normalized to [-1, 1].
    Default robosuite scaling: 0.05 m per unit position, 0.5 rad per unit rotation.

    Steps env until EE within pos_tol meters of target_pos OR max_steps reached.
    Returns dict with steps_taken, final_ee_pos, reached, final_err.
    """
    target_pos = np.asarray(target_pos, dtype=np.float64)
    body_id = env.sim.model.body_name2id("gripper0_eef")
    grip_action = -1.0 if gripper_open else 1.0

    steps_taken = 0
    reached = False
    for step in range(max_steps):
        cur_ee = np.array(env.sim.data.body_xpos[body_id])
        err = target_pos - cur_ee
        err_norm = np.linalg.norm(err)
        if err_norm < pos_tol:
            reached = True
            break
        # Normalize position delta to [-1, 1]; rotations held at 0.
        action_pos = np.clip(err / osc_pos_scale, -1.0, 1.0)
        action = action_pos.tolist() + [0.0, 0.0, 0.0, grip_action]
        env.step(action)
        steps_taken = step + 1

    final_ee = np.array(env.sim.data.body_xpos[body_id])
    return {
        "steps": steps_taken,
        "final_ee_pos": final_ee.tolist(),
        "reached": reached,
        "final_err": float(np.linalg.norm(target_pos - final_ee)),
    }


# ---- Self-test (run as module): verify LocateBox + MoveTo on libero_object_swap t2 ----
if __name__ == "__main__":
    import pathlib
    import time
    from libero import benchmark, get_libero_path
    from libero.envs import OffScreenRenderEnv

    bd = benchmark.get_benchmark_dict()
    suite = bd["libero_object_swap"]()
    task = suite.get_task(2)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256, camera_widths=256, camera_depths=True,
    )
    env.seed(7); env.reset()
    init_states = suite.get_task_init_states(2)
    obs = env.set_init_state(init_states[0])
    # Wait a few steps for objects to settle (7-dim: 6 OSC delta + 1 gripper)
    for _ in range(10):
        obs, _, _, _ = env.step([0.0]*6 + [-1.0])

    print(f"task: {task.language}")
    print(f"EE pos start: {get_current_ee_pos(env)}")

    # --- TEST LOCATEBOX ---
    # Ground-truth salad_dressing pos (privileged — only used for sanity)
    gt_pos = np.array(env.sim.data.body_xpos[env.sim.model.body_name2id("salad_dressing_1_main")])
    print(f"ground-truth salad_dressing pos: {gt_pos}")

    # Project gt_pos into LLM-view pixel using robosuite's helper
    px, py = project_world_to_pixel(env, gt_pos, camera="agentview", llm_view_rotated=True)
    print(f"projected LLM-view pixel: ({px}, {py})")

    # Test unprojection: 20x20 bbox around the projected pixel
    bbox = (px - 10, py - 10, px + 10, py + 10)
    depth_arr = obs["agentview_depth"]
    world_back = unproject_bbox_to_world(env, bbox, camera="agentview",
                                         depth_arr=depth_arr, llm_view_rotated=True)
    print(f"unprojected back from bbox: {world_back}")
    print(f"diff from gt: {np.linalg.norm(world_back - gt_pos):.4f} m")

    # --- TEST MOVE_EE_TO ---
    cur_ee = get_current_ee_pos(env)
    target = cur_ee + np.array([0.1, 0.0, -0.05])
    print(f"\n[MoveTo] from {cur_ee} -> {target}")
    t0 = time.time()
    result = move_ee_to(env, target, max_steps=200, pos_tol=0.02)
    print(f"[MoveTo] result: {result}, wall {time.time()-t0:.2f}s")

    env.close()
    print("DONE")