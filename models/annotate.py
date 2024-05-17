from ultralytics.data.annotator import auto_annotate
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage.draw import polygon
import os
import itertools
from skimage.measure import find_contours


class Annotator:
    def __init__(self, image_file):
        self.image_file = image_file
        self.base_file, _ = os.path.splitext(image_file)
        self.file_name = os.path.basename(self.base_file)
        self.output_dir = "annotations"
        self.det_model = r"models\object_detection_model.pt"
        self.sam_model = r"models\mobile_sam.pt"
        self.image = self.load_image()
        self.original_shape = self.image.shape[:2]
        self.mask = None

    def perform_auto_annotation(self):
        auto_annotate(
            data=self.image_file,
            det_model=self.det_model,
            sam_model=self.sam_model,
            output_dir=self.output_dir,
        )

    def load_image(self):
        return mpimg.imread(self.image_file)

    def parse_annotation_data(self):
        annotation_file = os.path.join(self.output_dir, f"{self.file_name}.txt")
        with open(annotation_file, "r") as file:
            data = file.read()
            segments = data.split(" ")[1:]  # Skip the first value which is usually the class ID
        return np.array(segments, dtype=float).reshape(-1, 2)

    def create_mask(self, segments):
        mask = np.zeros(self.original_shape)
        rr, cc = polygon(
            segments[:, 1] * self.original_shape[0],
            segments[:, 0] * self.original_shape[1],
        )
        mask[rr, cc] = 1
        return mask

    def apply_mask(self):
        masked_image = np.zeros_like(self.image)
        masked_image[:, :, 0] = np.where(
            self.mask == 1, 0, self.image[:, :, 0]
        )  # Set red channel to 0
        masked_image[:, :, 1] = np.where(
            self.mask == 1, 0, self.image[:, :, 1]
        )  # Set green channel to 0
        masked_image[:, :, 2] = np.where(
            self.mask == 1, 255, self.image[:, :, 2]
        )  # Set blue channel to 255
        return masked_image

    def plot_and_save_image(self, masked_image, output_file):
        plt.figure(
            figsize=(self.original_shape[1] / 100, self.original_shape[0] / 100)
        )  # Set the figure size based on original image dimensions
        plt.imshow(self.image)
        plt.imshow(
            masked_image,
            alpha=0.25,
            extent=[0, self.original_shape[1], self.original_shape[0], 0],
        )  # Set transparency to 0.25 and extend to original image dimensions
        plt.axis("off")  # Remove the axis
        plt.tight_layout()  # Ensure the plot is tightly arranged
        plt.savefig(
            output_file, bbox_inches="tight", pad_inches=0, dpi=100
        )  # Save the figure with tight bounding box and appropriate dpi
        plt.close()  # Close the figure to free up memory

    def annotate_and_mask(self):
        # Perform auto annotation
        self.perform_auto_annotation()

        # Load and parse the annotation data
        segments = self.parse_annotation_data()

        # Create mask
        self.mask = self.create_mask(segments)

        # Apply mask to the original image
        masked_image = self.apply_mask()

        # Plot and save the masked image
        output_file = os.path.join("masked", f"masked-{self.file_name}.jpg")
        self.plot_and_save_image(masked_image, output_file)
        return masked_image

    def area(self):
        if self.mask is None:
            raise ValueError("No mask found. Please run annotate_and_mask() first.")
        return np.sum(self.mask)

    def centroid(self):
        y_coords, x_coords = np.where(self.mask == 1)
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
        return centroid_y, centroid_x

    def distance_from_centroid(self, y, x, centroid_y, centroid_x):
        return np.sqrt((y - centroid_y) ** 2 + (x - centroid_x) ** 2)

    def length(self):
        if self.mask is None:
            raise ValueError("No mask found. Please run annotate_and_mask() first.")

        # Calculate centroid of the mask
        centroid_y, centroid_x = self.centroid()

        # Find the lengths of horizontal lines inside the masked polygon
        line_lengths = []
        for y, row in enumerate(self.mask):
            inside = False
            start_x = 0
            for x, value in enumerate(row):
                if value == 1 and not inside:
                    inside = True
                    start_x = x
                elif value == 0 and inside:
                    inside = False
                    end_x = x
                    length = end_x - start_x
                    distance = self.distance_from_centroid(y, (start_x + end_x) / 2, centroid_y, centroid_x)
                    line_lengths.append((length, distance))
            if inside:
                length = len(row) - start_x
                distance = self.distance_from_centroid(y, (start_x + len(row)) / 2, centroid_y, centroid_x)
                line_lengths.append((length, distance))

        # Sort lengths based on distance from centroid and find the top 50 longest lines
        top_lengths = sorted(line_lengths, key=lambda x: (x[1], -x[0]))[:50]

        # Calculate the average length
        if len(top_lengths) > 0:
            average_length = sum([length for length, _ in top_lengths]) / len(top_lengths)
        else:
            average_length = 0

        # Store the top lengths for visualization
        self.top_lengths = top_lengths

        return average_length

    def width(self):
        if self.mask is None:
            raise ValueError("No mask found. Please run annotate_and_mask() first.")

        # Calculate centroid of the mask
        centroid_y, centroid_x = self.centroid()

        # Find the lengths of vertical lines inside the masked polygon
        line_lengths = []
        for x, col in enumerate(self.mask.T):
            inside = False
            start_y = 0
            for y, value in enumerate(col):
                if value == 1 and not inside:
                    inside = True
                    start_y = y
                elif value == 0 and inside:
                    inside = False
                    end_y = y
                    length = end_y - start_y
                    distance = self.distance_from_centroid((start_y + end_y) / 2, x, centroid_y, centroid_x)
                    line_lengths.append((length, distance))
            if inside:
                length = len(col) - start_y
                distance = self.distance_from_centroid((start_y + len(col)) / 2, x, centroid_y, centroid_x)
                line_lengths.append((length, distance))

        # Sort lengths based on distance from centroid and find the top 50 longest lines
        top_lengths = sorted(line_lengths, key=lambda x: (x[1], -x[0]))[:50]

        # Calculate the average width
        if len(top_lengths) > 0:
            average_width = sum([length for length, _ in top_lengths]) / len(top_lengths)
        else:
            average_width = 0

        # Store the top widths for visualization
        self.top_widths = top_lengths

        return average_width

    def perimeter(self):
        if self.mask is None:
            raise ValueError("No mask found. Please run annotate_and_mask() first.")

        # Find contours of the mask
        contours = find_contours(self.mask, level=0.5)

        # Calculate the perimeter by summing the lengths of the contours
        perimeter = 0
        for contour in contours:
            perimeter += np.sum(np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1)))

        return perimeter

    def visualize(self):
        if self.mask is None:
            raise ValueError("No mask found. Please run annotate_and_mask() first.")

        # Ensure length() and width() have been called to get the top 50 longest lines
        if not hasattr(self, "top_lengths"):
            self.length()
        if not hasattr(self, "top_widths"):
            self.width()

        # Create a copy of the image to draw lines on
        vis_image = self.image.copy()

        # Find horizontal lines inside the masked polygon
        line_lengths = []
        for y, row in enumerate(self.mask):
            inside = False
            start_x = 0
            for x, value in enumerate(row):
                if value == 1 and not inside:
                    inside = True
                    start_x = x
                elif value == 0 and inside:
                    inside = False
                    end_x = x
                    length = end_x - start_x
                    distance = self.distance_from_centroid(y, (start_x + end_x) / 2, *self.centroid())
                    line_lengths.append((y, start_x, end_x, length, distance))
            if inside:
                length = len(row) - start_x
                distance = self.distance_from_centroid(y, (start_x + len(row)) / 2, *self.centroid())
                line_lengths.append((y, start_x, len(row), length, distance))

        # Sort lengths based on distance from centroid and find the top 50 longest horizontal lines
        top_horizontal_lengths = sorted(line_lengths, key=lambda x: (x[4], -x[3]))[:50]

        # Draw the top 50 longest horizontal lines
        for y, start_x, end_x, length, distance in top_horizontal_lengths:
            vis_image[y, start_x:end_x] = [255, 0, 0]  # Draw the line in red color

        # Find vertical lines inside the masked polygon
        line_lengths = []
        for x, col in enumerate(self.mask.T):
            inside = False
            start_y = 0
            for y, value in enumerate(col):
                if value == 1 and not inside:
                    inside = True
                    start_y = y
                elif value == 0 and inside:
                    inside = False
                    end_y = y
                    length = end_y - start_y
                    distance = self.distance_from_centroid((start_y + end_y) / 2, x, *self.centroid())
                    line_lengths.append((x, start_y, end_y, length, distance))
            if inside:
                length = len(col) - start_y
                distance = self.distance_from_centroid((start_y + len(col)) / 2, x, *self.centroid())
                line_lengths.append((x, start_y, len(col), length, distance))

        # Sort lengths based on distance from centroid and find the top 50 longest vertical lines
        top_vertical_lengths = sorted(line_lengths, key=lambda x: (x[4], -x[3]))[:50]

        # Draw the top 50 longest vertical lines
        for x, start_y, end_y, length, distance in top_vertical_lengths:
            vis_image[start_y:end_y, x] = [0, 0, 255]  # Draw the line in blue color

        # Save the visualized image
        output_file = os.path.join("masked", f"visualized-{self.file_name}.jpg")
        plt.figure(figsize=(self.original_shape[1] / 100, self.original_shape[0] / 100))
        plt.imshow(vis_image)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close()
        print(f"Visualization saved to {output_file}")
