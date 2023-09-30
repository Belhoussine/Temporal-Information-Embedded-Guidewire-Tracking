import os
import cv2
import numpy as np
from src.utils.dataset import mask_to_bbox

# Calculate the union of bounding boxes
def union_bboxes(bboxes):
    if len(bboxes) == 0:
        return None
    x1 = min([bbox[0] for bbox in bboxes])
    y1 = min([bbox[1] for bbox in bboxes])
    x2 = max([bbox[2] for bbox in bboxes])
    y2 = max([bbox[3] for bbox in bboxes])
    return [x1, y1, x2, y2]

# Create new datasets with modified frames
def process_dataset(input_path, focused_path, cropped_path):
    # Iterate through each sequence folder in the input dataset
    for sequence_folder in os.listdir(input_path):
        sequence_path = os.path.join(input_path, sequence_folder)

        if not os.path.isdir(sequence_path):
            continue

        # Initialize a list to store bounding boxes of frames in this sequence
        frame_bboxes = []

        # Iterate through frames in the 'gt' folder
        gt_folder = os.path.join(sequence_path, 'gt')
        for mask_file in os.listdir(gt_folder):
            if mask_file.endswith('.png'):
                mask_path = os.path.join(gt_folder, mask_file)

                # Calculate the bounding box using mask_to_bbox function
                bbox = mask_to_bbox(mask_path, is_dhm=False)

                if bbox is not None:
                    frame_bboxes.append(bbox)

        # Calculate the global bounding box for this sequence
        global_bbox = union_bboxes(frame_bboxes)

        # Create the output folder structure for modified frames
        output_focused_path = os.path.join(focused_path, sequence_folder)
        os.makedirs(output_focused_path, exist_ok=True)
        os.makedirs(os.path.join(output_focused_path, 'png'), exist_ok=True)
        os.makedirs(os.path.join(output_focused_path, 'gt'), exist_ok=True)

        # Create the output folder structure for cropped square images
        output_cropped_path = os.path.join(cropped_path,sequence_folder)
        os.makedirs(output_cropped_path, exist_ok=True)
        os.makedirs(os.path.join(output_cropped_path, 'png'), exist_ok=True)
        os.makedirs(os.path.join(output_cropped_path, 'gt'), exist_ok=True)

        # Iterate through frames in the 'png' folder and modify them
        png_folder = os.path.join(sequence_path, 'png')
        for frame_file in os.listdir(png_folder):
            if frame_file.endswith('.png'):
                frame_path = os.path.join(png_folder, frame_file)
                frame = cv2.imread(frame_path)

                if global_bbox is not None:
                    # Create a mask for the global bounding box area
                    bbox_mask = np.zeros_like(frame)
                    x1, y1, x2, y2 = global_bbox
                    bbox_mask[y1:y2, x1:x2] = 255

                    # Apply the mask to the frame to black out pixels outside the bounding box
                    focused_frame = cv2.bitwise_and(frame, bbox_mask)

                    # Crop the image as a square based on the biggest side of the bounding box
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    max_side = max(bbox_width, bbox_height)
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2
                    x1_cropped = x_center - max_side // 2
                    y1_cropped = y_center - max_side // 2
                    x2_cropped = x1_cropped + max_side
                    y2_cropped = y1_cropped + max_side
                    cropped_frame = frame[y1_cropped:y2_cropped, x1_cropped:x2_cropped]

                    # Crop the corresponding mask
                    mask_path = os.path.join(gt_folder, frame_file)
                    mask = cv2.imread(mask_path)
                    cropped_mask = mask[y1_cropped:y2_cropped, x1_cropped:x2_cropped]

                # Save the focused frame to the focused dataset folder
                if not focused_frame is None and not focused_frame.all():
                    # output_focused_frame_path = os.path.join(output_focused_path, frame_file)
                    output_focused_frame_path = os.path.join(output_focused_path, 'png', frame_file)
                    output_focused_mask_path = os.path.join(output_focused_path, 'gt', frame_file)
                    cv2.imwrite(output_focused_frame_path, focused_frame)
                    cv2.imwrite(output_focused_mask_path, mask)
                    print(f"Saved {output_focused_frame_path}")

                # Save the cropped square image to the cropped dataset folder
                if not cropped_frame is None and np.any(cropped_frame > 0):
                    # output_cropped_frame_path = os.path.join(output_cropped_path, frame_file)
                    output_cropped_frame_path = os.path.join(output_cropped_path, 'png', frame_file)
                    output_cropped_mask_path = os.path.join(output_cropped_path, 'gt', frame_file)
                    cv2.imwrite(output_cropped_frame_path, cropped_frame)
                    cv2.imwrite(output_cropped_mask_path, cropped_mask)
                    print(f"Saved cropped: {output_cropped_path}")
                print()


if __name__ == "__main__":
    input_folder = '/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets/Guide_wire_sythetic_data'
    
    # Ensure the output folders exist
    focused_folder = '/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets/synth_guidewire_focused_data'
    os.makedirs(focused_folder, exist_ok=True)
    
    cropped_folder = '/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets/synth_guidewire_cropped_data'
    os.makedirs(cropped_folder, exist_ok=True)
    
    # Process the dataset
    process_dataset(input_folder, focused_folder, cropped_folder)
