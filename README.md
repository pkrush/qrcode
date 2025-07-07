# Image Point Editor

A Python GUI application for perspective correction and image warping with fine-tuning capabilities.

## Features

- **4-Point Perspective Transform**: Select 4 corner points to define a region for perspective correction
- **Automatic Square Warping**: Converts selected region to a perfect square with dimensions divisible by grid cells
- **Grid Overlay**: Visualize the transformed region with customizable grid divisions
- **Fine Warp Adjustment**: Apply additional local corrections using point pairs after initial transform
- **Persistent Corner Points**: Automatically saves and loads corner selections between sessions

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have Python 3.x installed (tested with Python 3.8+)

## Usage

1. Run the application:
```bash
python3 image_editor.py
```

2. Place your image file named `image_720.png` in the same directory as the script

3. **Basic Workflow**:
   - Click 4 corner points on the image to define the region
   - Click "Warp to Square" to create a perfect square
   - Click "Transform Image" to apply perspective correction
   - Optionally use "Show Grid" to visualize grid divisions

4. **Fine Warping** (for additional adjustments):
   - After transform, click "Start Fine Warp"
   - Click pairs of points: source (problem area) → destination (corrected position)
   - Click "Apply Fine Warp" when done

## Controls

- **Grid Cells**: Number of grid divisions (default: 29)
- **Reset**: Clear all points and start over
- **Show/Hide Grid**: Toggle grid overlay on the square region

## Files

- `image_editor.py` - Main application code
- `corner_points.json` - Automatically saved corner point selections
- `requirements.txt` - Python dependencies

## Point Labeling

- Points are numbered 1-4 in the order clicked
- During dragging, selected point turns yellow
- Fine warp points: purple (anchors), red (source), green (destination)

## Technical Details

- Images are automatically converted to grayscale
- Square dimensions are made divisible by grid cells value
- Uses OpenCV for perspective transforms
- Moving Least Squares (MLS) algorithm for fine warping