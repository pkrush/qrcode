try:
    # Python 3
    import tkinter as tk
    from tkinter import Canvas, Button, Frame, Label, Entry
except ImportError:
    # Python 2
    import Tkinter as tk
    from Tkinter import Canvas, Button, Frame, Label, Entry
from PIL import Image, ImageTk
import os
import math
import numpy as np
import cv2
import json


class ImageEditor:
    def __init__(self, master, line_width=1):
        self.master = master
        self.master.title("Image Point Editor")

        # Create main frame
        self.main_frame = Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create button frame
        self.button_frame = Frame(self.main_frame)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        # Create warp button
        self.warp_button = Button(self.button_frame, text="Warp to Square",
                                  command=self.warp_to_square, state=tk.DISABLED)
        self.warp_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Create transform image button
        self.transform_button = Button(self.button_frame, text="Transform Image",
                                       command=self.transform_image, state=tk.DISABLED)
        self.transform_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Create reset button
        self.reset_button = Button(self.button_frame, text="Reset",
                                   command=self.reset_all)
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add grid controls
        Label(self.button_frame, text="Grid Cells:").pack(side=tk.LEFT, padx=(20, 5))
        self.grid_cells_entry = Entry(self.button_frame, width=5)
        self.grid_cells_entry.insert(0, "29")
        self.grid_cells_entry.pack(side=tk.LEFT, padx=5)

        # Create show grid button
        self.grid_button = Button(self.button_frame, text="Show Grid",
                                  command=self.toggle_grid)
        self.grid_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add fine warp controls
        self.fine_warp_button = Button(self.button_frame, text="Start Fine Warp",
                                       command=self.start_fine_warp, state=tk.DISABLED)
        self.fine_warp_button.pack(side=tk.LEFT, padx=(20, 5), pady=5)

        self.apply_fine_warp_button = Button(self.button_frame, text="Apply Fine Warp",
                                             command=self.apply_fine_warp, state=tk.DISABLED)
        self.apply_fine_warp_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Status label
        self.status_label = Label(self.button_frame, text="Draw 4 corner points", fg="blue")
        self.status_label.pack(side=tk.LEFT, padx=(20, 5))

        # Canvas for displaying image and drawing
        self.canvas = Canvas(self.main_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Points and lines data
        self.points = []
        self.point_objects = []
        self.line_objects = []
        self.selected_point = None
        self.dragging = False
        self.line_width = line_width
        self.original_image = None
        self.warped_image = None
        self.grid_lines = []
        self.grid_shown = False

        # Fine warp data
        self.fine_warp_mode = False
        self.fine_warp_pairs = []
        self.current_source_point = None
        self.picking_destination = False
        self.fine_warp_objects = []

        # Points file
        self.points_file = "corner_points.json"

        # Load and display image
        self.load_image()

        # Try to load saved points
        self.load_points()

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def load_image(self):
        try:
            # Load the image
            image_path = "image_720.png"
            if os.path.exists(image_path):
                # Load and convert to grayscale
                color_image = Image.open(image_path)
                self.image = color_image.convert('L')  # Convert to grayscale
                self.original_image = np.array(self.image)
                self.photo = ImageTk.PhotoImage(self.image)

                # Update canvas size to match image
                self.canvas.config(width=self.image.width, height=self.image.height)

                # Display the image
                self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            else:
                print("Image file '{}' not found!".format(image_path))
                # Create a placeholder canvas
                self.canvas.config(width=720, height=480)
                self.canvas.create_text(360, 240, text="image_720.png not found",
                                        font=("Arial", 24), fill="white")
        except Exception as e:
            print("Error loading image: {}".format(e))
            self.canvas.config(width=720, height=480)
            self.canvas.create_text(360, 240, text="Error loading image",
                                    font=("Arial", 24), fill="white")

    def on_click(self, event):
        if self.fine_warp_mode:
            self.handle_fine_warp_click(event.x, event.y)
            return

        # Check if clicking on existing point
        clicked_point = None
        min_distance = float('inf')

        # Find the closest point within threshold
        for i, (px, py) in enumerate(self.points):
            distance = math.sqrt((event.x - px) ** 2 + (event.y - py) ** 2)
            if distance < 10 and distance < min_distance:
                min_distance = distance
                clicked_point = i

        if clicked_point is not None:
            self.selected_point = clicked_point
            self.dragging = True
            return

        # Add new point if we have less than 4
        if len(self.points) < 4:
            self.add_point(event.x, event.y)

    def on_drag(self, event):
        if self.fine_warp_mode:
            return

        if self.dragging and self.selected_point is not None:
            # Update point position
            self.points[self.selected_point] = (event.x, event.y)
            self.redraw_all()

    def on_release(self, event):
        self.dragging = False
        self.selected_point = None

        # Save points after dragging if we have all 4
        if len(self.points) == 4:
            # Update original points when manually edited
            self.original_points = [p for p in self.points]
            self.save_points()

    def add_point(self, x, y):
        # Add point to list
        self.points.append((x, y))

        # Draw the point
        point_id = self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5,
                                           fill="red", outline="darkred", width=2)
        self.point_objects.append(point_id)

        # Draw lines connecting points
        if len(self.points) > 1:
            # Connect to previous point
            prev_x, prev_y = self.points[-2]
            line_id = self.canvas.create_line(prev_x, prev_y, x, y,
                                              fill="blue", width=self.line_width)
            self.line_objects.append(line_id)

        # If we have 4 points, close the shape
        if len(self.points) == 4:
            # Connect last point to first
            first_x, first_y = self.points[0]
            line_id = self.canvas.create_line(x, y, first_x, first_y,
                                              fill="blue", width=self.line_width)
            self.line_objects.append(line_id)
            # Enable warp button
            self.warp_button.config(state=tk.NORMAL)
            # Store original points for transformation
            self.original_points = [p for p in self.points]
            self.status_label.config(text="4 corners selected - Ready to warp")

            # Save the original corner points
            self.save_points()

    def redraw_all(self):
        # Delete all existing points and lines
        for obj in self.point_objects + self.line_objects:
            self.canvas.delete(obj)

        self.point_objects = []
        self.line_objects = []

        # Redraw all points
        for i, (x, y) in enumerate(self.points):
            # Highlight the selected point during dragging
            if self.dragging and i == self.selected_point:
                point_id = self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7,
                                                   fill="yellow", outline="orange", width=3)
            else:
                point_id = self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5,
                                                   fill="red", outline="darkred", width=2)
            self.point_objects.append(point_id)

            # Add point number label
            label_id = self.canvas.create_text(x + 10, y - 10, text=str(i + 1),
                                               fill="blue", font=("Arial", 10, "bold"))
            self.point_objects.append(label_id)

        # Redraw all lines
        if len(self.points) > 1:
            for i in range(len(self.points)):
                next_i = (i + 1) % len(self.points)
                if i < len(self.points) - 1 or len(self.points) == 4:
                    x1, y1 = self.points[i]
                    x2, y2 = self.points[next_i]
                    line_id = self.canvas.create_line(x1, y1, x2, y2,
                                                      fill="blue", width=self.line_width)
                    self.line_objects.append(line_id)

    def warp_to_square(self):
        if len(self.points) != 4:
            return

        # Get grid cells value
        try:
            grid_cells = int(self.grid_cells_entry.get())
            if grid_cells <= 0:
                grid_cells = 16
        except ValueError:
            grid_cells = 16

        # Calculate center of the 4 points
        center_x = sum(p[0] for p in self.points) / 4
        center_y = sum(p[1] for p in self.points) / 4

        # Calculate average distance from center
        avg_distance = sum(math.sqrt((p[0] - center_x) ** 2 + (p[1] - center_y) ** 2)
                           for p in self.points) / 4

        # Calculate side length and make it divisible by grid_cells
        raw_side = avg_distance * 2 / math.sqrt(2)
        side_length = int(raw_side / grid_cells) * grid_cells

        # Ensure minimum size
        if side_length < grid_cells:
            side_length = grid_cells

        half_side = side_length / 2

        # Set new square positions
        self.points[0] = (center_x - half_side, center_y - half_side)  # top-left
        self.points[1] = (center_x + half_side, center_y - half_side)  # top-right
        self.points[2] = (center_x + half_side, center_y + half_side)  # bottom-right
        self.points[3] = (center_x - half_side, center_y + half_side)  # bottom-left

        # Redraw everything
        self.redraw_all()

        # Enable transform button
        self.transform_button.config(state=tk.NORMAL)
        self.status_label.config(text="Square created - Ready to transform")

    def transform_image(self):
        if self.original_image is None or len(self.points) != 4:
            return

        # Convert points to numpy arrays
        src_pts = np.float32(self.original_points)
        dst_pts = np.float32(self.points)

        # Calculate homography matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Get output image dimensions
        height, width = self.original_image.shape[:2]

        # Apply perspective transform
        self.warped_image = cv2.warpPerspective(self.original_image, matrix, (width, height))

        # Convert back to PIL Image and display
        if len(self.warped_image.shape) == 3:
            # Color image
            warped_pil = Image.fromarray(cv2.cvtColor(self.warped_image, cv2.COLOR_BGR2RGB))
        else:
            # Grayscale image
            warped_pil = Image.fromarray(self.warped_image)

        self.photo = ImageTk.PhotoImage(warped_pil)
        self.canvas.itemconfig(self.image_on_canvas, image=self.photo)

        # Enable fine warp button
        self.fine_warp_button.config(state=tk.NORMAL)
        self.status_label.config(text="Transform complete - Fine warp available")

    def reset_all(self):
        # Clear all points and lines
        for obj in self.point_objects + self.line_objects:
            self.canvas.delete(obj)

        self.points = []
        self.point_objects = []
        self.line_objects = []
        self.selected_point = None
        self.dragging = False

        # Disable buttons
        self.warp_button.config(state=tk.DISABLED)
        self.transform_button.config(state=tk.DISABLED)

        # Restore original image
        if self.original_image is not None:
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.itemconfig(self.image_on_canvas, image=self.photo)

        # Clear grid if shown
        self.clear_grid()

        # Reset fine warp
        self.clear_fine_warp()
        self.fine_warp_mode = False
        self.fine_warp_button.config(state=tk.DISABLED)
        self.apply_fine_warp_button.config(state=tk.DISABLED)
        self.status_label.config(text="Draw 4 corner points")

    def toggle_grid(self):
        if self.grid_shown:
            self.clear_grid()
        else:
            self.show_grid()

    def clear_grid(self):
        for line in self.grid_lines:
            self.canvas.delete(line)
        self.grid_lines = []
        self.grid_shown = False
        self.grid_button.config(text="Show Grid")

    def show_grid(self):
        if len(self.points) != 4:
            return

        # Clear existing grid
        self.clear_grid()

        # Get grid cells value
        try:
            grid_cells = int(self.grid_cells_entry.get())
            if grid_cells <= 0:
                grid_cells = 16
        except ValueError:
            grid_cells = 16

        # Get square bounds
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Draw vertical lines (grid_cells + 1 lines)
        for i in range(grid_cells + 1):
            x = min_x + (max_x - min_x) * i / grid_cells
            line = self.canvas.create_line(x, min_y, x, max_y,
                                           fill="green", width=1)
            self.grid_lines.append(line)

        # Draw horizontal lines (grid_cells + 1 lines)
        for i in range(grid_cells + 1):
            y = min_y + (max_y - min_y) * i / grid_cells
            line = self.canvas.create_line(min_x, y, max_x, y,
                                           fill="green", width=1)
            self.grid_lines.append(line)

        self.grid_shown = True
        self.grid_button.config(text="Hide Grid")

    def start_fine_warp(self):
        self.fine_warp_mode = True
        self.fine_warp_pairs = []

        # Add the 4 corner points as fixed anchors (src = dst)
        if len(self.points) == 4:
            for point in self.points:
                self.fine_warp_pairs.append((point, point))

        self.current_source_point = None
        self.picking_destination = False
        self.clear_fine_warp()

        # Draw the corner anchor points
        for src, dst in self.fine_warp_pairs:
            # Draw anchor points in purple - tiny 1 pixel
            point_id = self.canvas.create_oval(src[0] - 1, src[1] - 1, src[0] + 1, src[1] + 1,
                                               fill="purple", outline="purple", width=1)
            self.fine_warp_objects.append(point_id)

        self.update_fine_warp_status()
        self.fine_warp_button.config(text="Cancel Fine Warp")

    def clear_fine_warp(self):
        for obj in self.fine_warp_objects:
            self.canvas.delete(obj)
        self.fine_warp_objects = []

    def handle_fine_warp_click(self, x, y):
        if not self.picking_destination:
            # Picking source point
            self.current_source_point = (x, y)
            # Draw source point (red) - tiny 1 pixel
            point_id = self.canvas.create_oval(x - 1, y - 1, x + 1, y + 1,
                                               fill="red", outline="red", width=1)
            self.fine_warp_objects.append(point_id)
            self.picking_destination = True
            self.update_fine_warp_status()
        else:
            # Picking destination point
            dest_point = (x, y)
            # Draw destination point (green) - tiny 1 pixel
            point_id = self.canvas.create_oval(x - 1, y - 1, x + 1, y + 1,
                                               fill="green", outline="green", width=1)
            self.fine_warp_objects.append(point_id)

            # Draw arrow from source to destination - thin line
            arrow_id = self.canvas.create_line(self.current_source_point[0],
                                               self.current_source_point[1],
                                               x, y, fill="blue", width=1,
                                               arrow=tk.LAST, arrowshape=(4, 5, 2))
            self.fine_warp_objects.append(arrow_id)

            # Store the pair
            self.fine_warp_pairs.append((self.current_source_point, dest_point))

            # Reset for next pair
            self.current_source_point = None
            self.picking_destination = False
            self.update_fine_warp_status()

            # Enable apply button (we already have 4 corner points + at least 1 user point)
            self.apply_fine_warp_button.config(state=tk.NORMAL)

    def update_fine_warp_status(self):
        if not self.fine_warp_mode:
            return

        # Subtract 4 for the corner anchors when showing count
        user_pairs = len(self.fine_warp_pairs) - 4

        if self.picking_destination:
            status = "Click destination for point {}".format(user_pairs + 1)
        else:
            if user_pairs > 0:
                status = "Click source point {} (or Apply when ready)".format(user_pairs + 1)
            else:
                status = "Click source point to adjust (corners are anchored)"

        self.status_label.config(text=status)

        # Update button text
        if self.fine_warp_mode:
            self.fine_warp_button.config(command=self.cancel_fine_warp)

    def cancel_fine_warp(self):
        self.fine_warp_mode = False
        self.clear_fine_warp()
        self.fine_warp_pairs = []
        self.current_source_point = None
        self.picking_destination = False
        self.fine_warp_button.config(text="Start Fine Warp", command=self.start_fine_warp)
        self.apply_fine_warp_button.config(state=tk.DISABLED)
        self.status_label.config(text="Fine warp cancelled")

    def apply_fine_warp(self):
        # Need at least 4 corners + 1 user point = 5 pairs minimum
        if len(self.fine_warp_pairs) < 5 or self.warped_image is None:
            return

        # Convert pairs to numpy arrays
        src_points = np.float32([p[0] for p in self.fine_warp_pairs])
        dst_points = np.float32([p[1] for p in self.fine_warp_pairs])

        # Get current image to warp
        if self.warped_image is not None:
            current_image = self.warped_image.copy()
        else:
            current_image = self.original_image.copy()

        height, width = current_image.shape[:2]

        # Use Moving Least Squares (MLS) deformation - simple implementation
        result = self.apply_mls_warp(current_image, src_points, dst_points)

        # Update the warped image
        self.warped_image = result

        # Convert back to PIL and display
        if len(result.shape) == 3:
            warped_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            warped_pil = Image.fromarray(result)

        self.photo = ImageTk.PhotoImage(warped_pil)
        self.canvas.itemconfig(self.image_on_canvas, image=self.photo)

        # Clear fine warp mode
        self.cancel_fine_warp()
        # Show user pairs count (excluding the 4 corner anchors)
        user_pairs = len(self.fine_warp_pairs) - 4
        self.status_label.config(text="Fine warp applied - {} adjustment points used".format(user_pairs))

    @staticmethod
    def apply_mls_warp(image, src_pts, dst_pts):
        """Apply Moving Least Squares deformation"""
        height, width = image.shape[:2]

        # Create a grid of coordinates
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)

        # Calculate displacement for each grid point
        displacements = np.zeros_like(grid_points)

        for i, grid_pt in enumerate(grid_points):
            # Calculate weights based on distance to control points
            distances = np.linalg.norm(src_pts - grid_pt, axis=1)
            # Avoid division by zero
            distances = np.maximum(distances, 1e-6)
            # Use inverse distance weighting
            weights = 1.0 / (distances ** 2)
            weights = weights / np.sum(weights)

            # Calculate weighted displacement
            displacement = np.sum(weights[:, np.newaxis] * (dst_pts - src_pts), axis=0)
            displacements[i] = displacement

        # Create displacement maps
        map_x = (grid_points[:, 0] + displacements[:, 0]).reshape(height, width).astype(np.float32)
        map_y = (grid_points[:, 1] + displacements[:, 1]).reshape(height, width).astype(np.float32)

        # Apply remapping
        result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

        return result

    def save_points(self):
        """Save the original 4 corner points to JSON file"""
        if len(self.points) == 4 and hasattr(self, 'original_points'):
            data = {
                'corner_points': [(float(x), float(y)) for x, y in self.original_points],
                'grid_cells': self.grid_cells_entry.get()
            }
            try:
                with open(self.points_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print("Original corner points saved to {}".format(self.points_file))
            except Exception as e:
                print("Error saving points: {}".format(e))

    def load_points(self):
        """Load saved corner points from JSON file"""
        if os.path.exists(self.points_file):
            try:
                with open(self.points_file, 'r') as f:
                    data = json.load(f)

                if 'corner_points' in data:
                    # Load the corner points
                    self.points = [(float(x), float(y)) for x, y in data['corner_points']]

                    # Update grid cells if available
                    if 'grid_cells' in data:
                        self.grid_cells_entry.delete(0, tk.END)
                        self.grid_cells_entry.insert(0, data['grid_cells'])

                    # Store as original points
                    self.original_points = [p for p in self.points]

                    # Redraw all points and lines
                    self.redraw_all()

                    # Enable warp button
                    self.warp_button.config(state=tk.NORMAL)
                    self.status_label.config(text="Loaded saved corner points - Ready to warp")

                    print("Corner points loaded from {}".format(self.points_file))
            except Exception as e:
                print("Error loading points: {}".format(e))


def main():
    root = tk.Tk()
    # You can change line_width here, e.g., ImageEditor(root, line_width=5)
    app = ImageEditor(root, line_width=1)
    root.mainloop()


if __name__ == "__main__":
    main()
