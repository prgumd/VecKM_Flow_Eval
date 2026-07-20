import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration: update these paths before running ---
SCENE_DIR = Path("../data/scenes/scene_03_04_000000")
OUTPUT_SCENE_DIR = Path("../data/scenes/scene03_40/03_40")
GT_FLOW_PATH = Path("../data/gt_flow/03_04_00_gt_flow_dist.npz")

data_t_path = SCENE_DIR / "dataset_events_t.npy"
data_xy_path = SCENE_DIR / "dataset_events_xy.npy"
data_p_path = SCENE_DIR / "dataset_events_p.npy"
event_npy_folder_path = OUTPUT_SCENE_DIR / "davis/left/events"
gt_save_path = OUTPUT_SCENE_DIR.parent / "03_40_gt.hdf5"
gt_npy_folder_path = OUTPUT_SCENE_DIR / "optical_flow"
gt_path = GT_FLOW_PATH
timestamps_txt_path = OUTPUT_SCENE_DIR / "timestamps_depth.txt"

# Ensure the directory exists
event_npy_folder_path.mkdir(parents=True, exist_ok=True)
gt_npy_folder_path.mkdir(parents=True, exist_ok=True)
#
gt_flow = np.load(gt_path)
#
timestamps = gt_flow['timestamps'][1:]
x_flow_dist = np.expand_dims(gt_flow['x_flow_dist'], axis=1)
y_flow_dist = np.expand_dims(gt_flow['y_flow_dist'], axis=1)
flow_dist = np.concatenate((x_flow_dist, y_flow_dist), axis=1)
print(flow_dist.shape,timestamps.shape)


xs = np.load(data_xy_path)[:,0]
ys = np.load(data_xy_path)[:,1]
ps = np.load(data_p_path)
ts = np.load(data_t_path)
print(xs.shape, ys.shape, ps.shape, ts.shape)
print(xs.max(), xs.min(), ys.max(), ys.min(), ts.max(), ts.min())
# print(ts)



for frame_index in range(0,len(timestamps)):
    flow = flow_dist[frame_index]
    # plt.imshow(flow[0], cmap='gray')
    # plt.show()
    np.save(os.path.join(gt_npy_folder_path, '{:06d}.npy'.format(frame_index)), flow)
    if frame_index == 0:
        t_start_idx = 0
    else:
        t_start = timestamps[frame_index-1]
        t_start_idx = np.searchsorted(ts, t_start, side='left')
    t_end = timestamps[frame_index]
    t_end_idx = np.searchsorted(ts, t_end, side='left')
    x = xs[t_start_idx:t_end_idx].astype(np.int16)
    y = ys[t_start_idx:t_end_idx].astype(np.int16)
    p = ps[t_start_idx:t_end_idx].astype(np.int8)
    t = ts[t_start_idx:t_end_idx].astype(np.float64)
    concatenated_data = np.stack((x, y, p, t), axis=-1)
    concatenated_dtype = [('x', '<i2'), ('y', '<i2'), ('p', 'i1'), ('ts', '<f8')]
    concatenated_data = np.core.records.fromarrays(concatenated_data.T, dtype=concatenated_dtype)

    # Open the HDF5 file
    with h5py.File(os.path.join(event_npy_folder_path, '{:06d}.h5'.format(frame_index)), 'w') as f:

        f.create_dataset('myDataset', data=concatenated_data)

np.savetxt(timestamps_txt_path, timestamps, fmt='%f')

# Open the HDF5 file and save the stacked_events array
with h5py.File(gt_save_path, 'w') as f:  # 'a' mode allows read/write if file exists
    if 'davis' not in f:
        davis_group = f.create_group('davis')
    else:
        davis_group = f['davis']

    if 'left' not in davis_group:
        left_group = davis_group.create_group('left')
    else:
        left_group = davis_group['left']

    if 'flow_dist' in left_group:
        del left_group['flow_dist']

    if 'flow_dist_ts' in left_group:
        del left_group['flow_dist_ts']

    # Delete the existing dataset if it exists

    left_group.create_dataset('flow_dist', data=flow_dist)
    left_group.create_dataset('flow_dist_ts', data=timestamps)