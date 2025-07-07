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
        
        # Load and display image
        self.load_image()
        
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
        # Check if clicking on existing point
        for i, (px, py) in enumerate(self.points):
            if abs(event.x - px) < 10 and abs(event.y - py) < 10:
                self.selected_point = i
                self.dragging = True
                return
        
        # Add new point if we have less than 4
        if len(self.points) < 4:
            self.add_point(event.x, event.y)
    
    def on_drag(self, event):
        if self.dragging and self.selected_point is not None:
            # Update point position
            self.points[self.selected_point] = (event.x, event.y)
            self.redraw_all()
    
    def on_release(self, event):
        self.dragging = False
        self.selected_point = None
    
    def add_point(self, x, y):
        # Add point to list
        self.points.append((x, y))
        
        # Draw the point
        point_id = self.canvas.create_oval(x-5, y-5, x+5, y+5, 
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
    
    def redraw_all(self):
        # Delete all existing points and lines
        for obj in self.point_objects + self.line_objects:
            self.canvas.delete(obj)
        
        self.point_objects = []
        self.line_objects = []
        
        # Redraw all points
        for x, y in self.points:
            point_id = self.canvas.create_oval(x-5, y-5, x+5, y+5, 
                                             fill="red", outline="darkred", width=2)
            self.point_objects.append(point_id)
        
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
        avg_distance = sum(math.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) 
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

def main():
    root = tk.Tk()
    # You can change line_width here, e.g., ImageEditor(root, line_width=5)
    app = ImageEditor(root, line_width=1)
    root.mainloop()

if __name__ == "__main__":
    main()