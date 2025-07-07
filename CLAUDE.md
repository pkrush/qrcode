# Claude Development Guide - Image Point Editor

## Project Overview

This is a tkinter-based GUI application for perspective correction and image warping. The main use case is correcting perspective distortion in images (particularly useful for QR codes or documents).

## Key Implementation Details

### Core Workflow
1. User selects 4 corner points on the image
2. Points are warped to form a perfect square (with dimensions divisible by grid_cells)
3. Perspective transform is applied to the entire image
4. Optional fine-tuning with additional control points

### Important Code Sections

**Point Management** (lines 134-177):
- Click detection uses closest point within 10-pixel threshold
- Points can be dragged to adjust position
- Original points are preserved separately from warped points

**Warping Logic** (lines 246-287):
- Square dimensions calculated to be divisible by grid_cells parameter
- Center point preserved during warping
- Uses `cv2.getPerspectiveTransform` for homography calculation

**Fine Warp System** (lines 402-533):
- Automatically includes 4 corners as fixed anchor points
- Uses Moving Least Squares (MLS) for smooth deformation
- Minimum 1 additional adjustment point required

### Data Persistence

**Saving** (lines 569-581):
- Saves `original_points` (not warped points) to `corner_points.json`
- Triggered on: initial 4-point completion, point dragging

**Loading** (lines 583-611):
- Loads on startup if JSON file exists
- Restores corner points and grid_cells setting

### Visual Feedback

- Regular points: 5px red circles with blue numbering
- Dragged point: 7px yellow/orange circle
- Fine warp markers: 1px dots (purple=anchors, red=source, green=destination)

## Common Modifications

### Change Default Grid Size
Edit line 48: `self.grid_cells_entry.insert(0, "29")`

### Adjust Point Selection Threshold
Edit line 146: Change `distance < 10` to desired pixel threshold

### Modify Line Width
Edit line 617: `app = ImageEditor(root, line_width=1)`

## Testing Commands

```bash
# Run the application
python3 image_editor.py

# Install dependencies
pip install -r requirements.txt
```

## Known Behaviors

1. Points are saved after each drag operation
2. Fine warp requires completed perspective transform first
3. Grid only displays when 4 points are defined
4. MLS warping is used instead of TPS due to OpenCV compatibility

## Debugging Tips

- Check console output for save/load confirmations
- Verify `corner_points.json` contains original (not warped) coordinates
- Point numbering helps identify selection order issues
- Purple dots in fine warp mode indicate corner anchors