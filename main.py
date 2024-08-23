import cupy as cp
import cupyx.scipy.ndimage as ndi
import cv2
import numpy as np
import torch
from tqdm import tqdm
import signal as sys_signal
import sys


class SeamCarver:
    def __init__(self, filename, protect_mask="", object_mask=""):
        # initialize parameter
        self.filename = filename

        # read in image and store as cp.float64 format
        self.in_image = cp.asarray(cv2.imread(filename).astype(cp.float64))
        self.in_height, self.in_width = self.in_image.shape[:2]

        # keep tracking resulting image
        self.out_image = cp.copy(self.in_image)

        # object removal --> self.object = True
        self.object = object_mask != ""
        if self.object:
            # read in object mask image file as cp.float64 format in gray scale
            self.mask = cp.asarray(cv2.imread(object_mask, 0).astype(cp.float64))
            self.protect = False
        # image re-sizing with or without protect mask
        else:
            self.protect = protect_mask != ""
            if self.protect:
                # if protect_mask filename is provided, read in protect mask image file as cp.float64 format in gray scale
                self.mask = cp.asarray(cv2.imread(protect_mask, 0).astype(cp.float64))

        # Define the Scharr
        self.scharr_x = cp.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=cp.float64)
        self.scharr_y = cp.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=cp.float64)

        self.kernel_x = cp.array([[0.0, 0.0, 0.0],
                                  [-1.0, 0.0, 1.0],
                                  [0.0, 0.0, 0.0]], dtype=cp.float64)

        self.kernel_y_left = cp.array([[0.0, 0.0, 0.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, -1.0, 0.0]], dtype=cp.float64)

        self.kernel_y_right = cp.array([[0.0, 0.0, 0.0],
                                        [1.0, 0.0, 0.0],
                                        [0.0, -1.0, 0.0]], dtype=cp.float64)

        # constant for covered area by protect mask or object mask
        self.constant = 1000

    def start(self):
        """
        :return:
        If object mask is provided --> object removal function will be executed
        else --> seam carving function (image retargeting) will be process
        """
        if self.object:
            self.object_removal()

    def object_removal(self):
        """
        :return:
        Object covered by mask will be removed first and seam will be inserted to return to original image dimension
        """
        rotate = False
        object_height, object_width = self.get_object_dimension()
        if object_height < object_width:
            self.out_image = self.rotate_image(self.out_image, 1)
            self.mask = self.rotate_mask(self.mask, 1)
            rotate = True

        total_seams = cp.sum(
            self.mask > 0
        ).item()  # Calculate total number of seams to remove

        with tqdm(total=total_seams, desc="Removing object seams", unit="seam") as pbar:
            while len(cp.where(self.mask[:, :] > 0)[0]) > 0:
                energy_map = self.calc_energy_map()
                energy_map[cp.where(self.mask[:, :] > 0)] *= -self.constant
                cumulative_map = self.cumulative_map_forward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                self.delete_seam(seam_idx)
                self.delete_seam_on_mask(seam_idx)
                pbar.update(1)  # Update progress bar after each seam removal

        if not rotate:
            num_pixels = self.in_width - self.out_image.shape[1]
        else:
            num_pixels = self.in_height - self.out_image.shape[1]

        self.seams_insertion(num_pixels)
        if rotate:
            self.out_image = self.rotate_image(self.out_image, 0)

    def seams_removal(self, num_pixel):
        if self.protect:
            for _ in range(num_pixel):
                energy_map = self.calc_energy_map()
                energy_map[cp.where(self.mask > 0)] *= self.constant
                cumulative_map = self.cumulative_map_forward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                self.delete_seam(seam_idx)
                self.delete_seam_on_mask(seam_idx)
        else:
            for _ in range(num_pixel):
                energy_map = self.calc_energy_map()
                cumulative_map = self.cumulative_map_forward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                self.delete_seam(seam_idx)

    def seams_insertion(self, num_pixel):
        if self.protect:
            temp_image = cp.copy(self.out_image)
            temp_mask = cp.copy(self.mask)
            seams_record = []

            for _ in range(num_pixel):
                energy_map = self.calc_energy_map()
                energy_map[cp.where(self.mask[:, :] > 0)] *= self.constant
                cumulative_map = self.cumulative_map_backward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                seams_record.append(seam_idx)
                self.delete_seam(seam_idx)
                self.delete_seam_on_mask(seam_idx)

            self.out_image = cp.copy(temp_image)
            self.mask = cp.copy(temp_mask)
            n = len(seams_record)
            for _ in range(n):
                seam = seams_record.pop(0)
                self.add_seam(seam)
                self.add_seam_on_mask(seam)
                seams_record = self.update_seams(seams_record, seam)
        else:
            temp_image = cp.copy(self.out_image)
            seams_record = []

            for _ in range(num_pixel):
                energy_map = self.calc_energy_map()
                cumulative_map = self.cumulative_map_backward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                seams_record.append(seam_idx)
                self.delete_seam(seam_idx)

            self.out_image = cp.copy(temp_image)
            n = len(seams_record)
            for _ in range(n):
                seam = seams_record.pop(0)
                self.add_seam(seam)
                seams_record = self.update_seams(seams_record, seam)

    def calc_energy_map(self):
        b, g, r = cp.split(self.out_image, 3, axis=2)
        grad_x_b = ndi.convolve(b[:, :, 0], self.scharr_x)
        grad_y_b = ndi.convolve(b[:, :, 0], self.scharr_y)
        grad_x_g = ndi.convolve(g[:, :, 0], self.scharr_x)
        grad_y_g = ndi.convolve(g[:, :, 0], self.scharr_y)
        grad_x_r = ndi.convolve(r[:, :, 0], self.scharr_x)
        grad_y_r = ndi.convolve(r[:, :, 0], self.scharr_y)
        return cp.abs(grad_x_b) + cp.abs(grad_y_b) + cp.abs(grad_x_g) + cp.abs(grad_y_g) + cp.abs(
            grad_x_r) + cp.abs(grad_y_r)

    def calc_neighbor_matrix(self, kernel):
        b, g, r = cp.split(self.out_image, 3, axis=2)
        return cp.abs(ndi.convolve(b[:, :, 0], kernel)) + cp.abs(ndi.convolve(g[:, :, 0], kernel)) + cp.abs(
            ndi.convolve(r[:, :, 0], kernel))

    def cumulative_map_backward(self, energy_map):
        m, n = energy_map.shape
        if m == 0 or n == 0:
            raise ValueError("Energy map dimensions must be greater than 0.")

        output = cp.copy(energy_map)

        # Shifted matrices for left and right neighbors
        left_shift = cp.pad(output[:-1, :-1], ((0, 0), (1, 0)), constant_values=cp.inf)
        right_shift = cp.pad(output[:-1, 1:], ((0, 0), (0, 1)), constant_values=cp.inf)
        up_shift = output[:-1, :]

        # Calculate minimum values across the rows
        min_cost = cp.minimum(cp.minimum(left_shift, right_shift), up_shift)

        # Update the cumulative map
        output[1:, :] += min_cost

        return output

    def cumulative_map_forward(self, energy_map):
        # Calculate neighbor matrices using the kernels
        matrix_x = self.calc_neighbor_matrix(self.kernel_x)
        matrix_y_left = self.calc_neighbor_matrix(self.kernel_y_left)
        matrix_y_right = self.calc_neighbor_matrix(self.kernel_y_right)

        m, n = energy_map.shape
        if m == 0 or n == 0:
            raise ValueError("Energy map dimensions must be greater than 0.")

        # Initialize the cumulative map with a copy of the energy map
        cumulative_map = cp.copy(energy_map)

        # Shifted cumulative map matrices for left, right, and up neighbors
        left_shift = cp.pad(cumulative_map[:-1, :-1], ((0, 0), (1, 0)), constant_values=cp.inf)
        right_shift = cp.pad(cumulative_map[:-1, 1:], ((0, 0), (0, 1)), constant_values=cp.inf)
        up_shift = cumulative_map[:-1, :]

        # Adjust neighbor matrices to match the dimensions of shifted matrices
        matrix_x = matrix_x[:-1, :]
        matrix_y_left = matrix_y_left[:-1, :]
        matrix_y_right = matrix_y_right[:-1, :]

        # Calculate the energy costs of moving from the left, right, and up
        e_left = left_shift + matrix_x + matrix_y_left
        e_right = right_shift + matrix_x + matrix_y_right
        e_up = up_shift + matrix_x

        # Calculate the minimum energy path for each pixel
        min_cost = cp.minimum(cp.minimum(e_left, e_right), e_up)

        # Update the cumulative map by adding the minimum cost path to the energy map
        cumulative_map[1:, :] += min_cost

        return cumulative_map

    def find_seam(self, cumulative_map):
        m, n = cumulative_map.shape

        # Initialize the seam with the minimum value from the last row
        seam = cp.zeros(m, dtype=cp.int32)
        seam[-1] = cp.argmin(cumulative_map[-1])

        # Calculate backtrack directions in a vectorized manner
        left_shift = cp.pad(cumulative_map[:-1, :-1], ((0, 0), (1, 0)), constant_values=cp.inf)
        right_shift = cp.pad(cumulative_map[:-1, 1:], ((0, 0), (0, 1)), constant_values=cp.inf)
        up_shift = cumulative_map[:-1, :]

        # Stack the shifts to find minimum
        direction_matrix = cp.stack([left_shift, up_shift, right_shift], axis=0)
        min_directions = cp.argmin(direction_matrix, axis=0) - 1  # -1, 0, 1 for left, up, right

        # Traverse the seam in reverse order
        seam_indices = cp.arange(m - 2, -1, -1)
        seam[seam_indices] = seam[seam_indices + 1] + min_directions[seam_indices, seam[seam_indices + 1]]

        return seam

    def delete_seam(self, seam_idx):
        m, n, _ = self.out_image.shape
        row_indices = cp.arange(m)

        # Create a boolean mask where seam indices will be removed
        mask = cp.ones((m, n), dtype=bool)
        mask[row_indices, seam_idx] = False

        # Apply the mask to remove the seam
        self.out_image = self.out_image[mask].reshape(m, n - 1, 3)

    def add_seam(self, seam_idx):
        m, n, c = self.out_image.shape
        row_indices = cp.arange(m)

        # Create an output array with one additional column
        output = cp.zeros((m, n + 1, c))

        # Identify where to insert the new seam and calculate its value
        left_neighbors = cp.maximum(seam_idx - 1, 0)

        left_vals = self.out_image[row_indices, left_neighbors, :]
        seam_vals = self.out_image[row_indices, seam_idx, :]
        new_seam_vals = (left_vals + seam_vals) / 2

        # Fill the new output array
        output[row_indices, :seam_idx, :] = self.out_image[row_indices, :seam_idx, :]
        output[row_indices, seam_idx, :] = new_seam_vals
        output[row_indices, seam_idx + 1 :, :] = self.out_image[
            row_indices, seam_idx:, :
        ]

        self.out_image = output

    def update_seams(self, remaining_seams, current_seam):
        remaining_seams = cp.array(remaining_seams)

        # Update the seam positions: add 2 to all elements where the seam is greater than or equal to current_seam
        remaining_seams += (remaining_seams >= current_seam).astype(cp.int32) * 2

        return remaining_seams.tolist()

    def rotate_image(self, image, ccw):
        if ccw:
            # Flip and then rotate the image 90 degrees counterclockwise
            return cp.rot90(cp.fliplr(image), 3)

        # Rotate the image 90 degrees clockwise
        return cp.rot90(image, 1)

    def rotate_mask(self, mask, ccw):
        if ccw:
            # Flip and then rotate the mask 90 degrees counterclockwise
            return cp.rot90(cp.fliplr(mask), 3)

        # Rotate the mask 90 degrees clockwise
        return cp.rot90(mask, 1)

    def delete_seam_on_mask(self, seam_idx):
        m, n = self.mask.shape
        mask_flat = self.mask.flatten()

        # Calculate the correct size after removing one column
        new_size = m * (n - 1)

        # Remove the seam from the mask
        seam_flat = cp.arange(m) * n + seam_idx
        mask_flat = cp.delete(mask_flat, seam_flat)

        # Ensure the reshaped array matches the expected dimensions
        self.mask = mask_flat[:new_size].reshape((m, n - 1))

    def add_seam_on_mask(self, seam_idx):
        m, n = self.mask.shape
        # Create an output array with one extra column for the new seam
        output = cp.zeros((m, n + 1))

        # Create a column index array for all rows
        col_indices = cp.arange(n)

        # Create a 2D array that has seam_idx for each row
        seam_insert_pos = seam_idx[:, None]

        # Create a boolean mask where new seams will be inserted
        insertion_mask = cp.arange(n + 1) == seam_insert_pos

        # Shift original columns by one where the insertion will happen
        mask_shifted = cp.where(
            insertion_mask, 0, col_indices + (col_indices >= seam_idx[:, None])
        )

        # Fill the output with the original mask values
        output[:, :-1] = self.mask[cp.arange(m)[:, None], mask_shifted]

        # Compute the new seam values as the average of neighboring pixels
        left_vals = cp.take(self.mask, cp.clip(seam_idx - 1, 0, n - 1))
        right_vals = cp.take(self.mask, cp.clip(seam_idx, 0, n - 1))
        new_seam_vals = (left_vals + right_vals) / 2

        # Place the new seam values in the output array
        output[cp.arange(m), seam_idx] = new_seam_vals

        self.mask = output

    def get_object_dimension(self):
        rows, cols = cp.where(self.mask > 0)
        height = cp.amax(rows) - cp.amin(rows) + 1
        width = cp.amax(cols) - cp.amin(cols) + 1
        return height, width

    def save_result(self, filename):
        cv2.imwrite(filename, cp.asnumpy(self.out_image).astype(cp.uint8))


def object_removal(filename_input, filename_output, filename_mask):
    obj = SeamCarver(filename_input, object_mask=filename_mask)
    try:
        obj.start()
    except KeyboardInterrupt as e:
        print(f"Error: {e}")
    finally:
        obj.save_result(filename_output)


def create_mask(
    filename_input, filename_output, show_detection=False, skip_coords=None
):
    if skip_coords is None:
        skip_coords = {(713, 878, 877, 1367)}

    image = cv2.imread(filename_input)
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    results = model(image)
    detections = results.xyxy[0].cpu().numpy()

    people_detections = [det for det in detections if int(det[5]) == 0]
    mask = np.zeros_like(image)

    for det in people_detections:
        x1, y1, x2, y2, conf, cls = det
        print(
            f"Person detected with confidence {conf:.2f}: Coordinates: ({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)})"
        )

        if (int(x1), int(y1), int(x2), int(y2)) in skip_coords or conf < 0.5:
            print("  Skipping from masking")
            continue

        mask[int(y1) : int(y2), int(x1) : int(x2)] = 255

        if show_detection:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)})",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    cv2.imwrite(filename_output, mask)
    if show_detection:
        cv2.imshow("Detections", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "C:\\Users\\Galahad\\Downloads\\IMG_1917.JPEG"
    output_image_path = "C:\\Users\\Galahad\\Downloads\\updated_IMG_1917.JPEG"
    mask_path = "C:\\Users\\Galahad\\Downloads\\mask.JPEG"

    create_mask(image_path, mask_path)
    object_removal(image_path, output_image_path, mask_path)