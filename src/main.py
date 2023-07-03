from utils.io import read_yaml
from models.robustPCA import RPCA
from utils.dataset import mask_to_bbox, stack_frames

RESOLUTION = (512, 512)


def main(frames, iterations, thresholds):
    datasets = {
        "SYNTH": [
            f"/home/belhoussine/dev/TUM/CS - Practical/Temporal-Information-Embedded-Guidewire-Tracking/datasets/Guide_wire_sythetic_data/sequence05_frame00050_sequence18_frame00100/png/navi000000{i}.png"
            for i in range(50, 50 + frames)
        ],
        "DHM": [
            f"/home/belhoussine/dev/TUM/CS - Practical/Temporal-Information-Embedded-Guidewire-Tracking/datasets/DHM-DATA/DHM_guidewiredataset/single_gw/2_data_slice61/{i}.png"
            for i in range(frames)
        ],
        "DHMv2": [
            f"/home/belhoussine/dev/TUM/CS - Practical/Temporal-Information-Embedded-Guidewire-Tracking/datasets/DHM-DATA_v2/DHM_guidewiredataset/single_gw/2_data_slice64/{i}.png"
            for i in range(frames)
        ],
    }
    for dataset, frame_paths in datasets.items():
        print(f"=> [Dataset] {dataset}")
        D = stack_frames(frame_paths, resolution=RESOLUTION,
                         n_frames=len(frame_paths))
        # Use RPCA to estimate the data as L + S, where L is low rank, and S is sparse
        rpca = RPCA(D)
        rpca.fit(max_iter=iterations, iter_print=100)
        # rpca.plot_results(resolution=RESOLUTION, thresh=thresholds)
        rpca.save_results(dataset, frames, iterations,
                          thresholds, resolution=RESOLUTION)


if __name__ == "__main__":
    config = read_yaml("../config.yaml")
    for frames in config["frames"]:
        for iterations in config["iterations"]:
            print(f"\n[Frames] {frames} - [Iterations] {iterations}")
            main(frames, iterations, config["thresholds"])
