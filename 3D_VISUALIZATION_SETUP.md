# 3D Tumor Visualization Setup

## Overview
The system now generates 3D surface meshes from tumor segmentation and displays them in an interactive Three.js viewer.

## Backend Implementation

### Dependencies Added
- `scikit-image` - For marching cubes algorithm
- `trimesh` - For mesh export (STL/OBJ)

### Files Created/Modified

1. **`server/utils/mesh_generation.py`** (NEW)
   - `extract_tumor_masks()` - Extracts binary masks for each subregion
   - `generate_mesh_from_mask()` - Creates 3D mesh using marching cubes
   - `export_mesh()` - Exports mesh as STL or OBJ
   - `generate_tumor_meshes()` - Main function to generate all meshes

2. **`server/app.py`** (MODIFIED)
   - Added mesh generation after inference
   - Uses ground truth segmentation (`seg.nii`) if available, otherwise uses predictions
   - Adds mesh URLs to API response
   - Updated download endpoint to serve STL files with CORS headers

### Mesh Generation Process

1. **Source Selection:**
   - If `seg.nii` (ground truth) is in ZIP → uses ground truth
   - Otherwise → uses predicted segmentation

2. **Label Mapping:**
   - Label 1 → Necrotic Tumor (NET) → Red mesh
   - Label 2 → Edema (ED) → Green mesh
   - Label 3 or 4 → Enhancing Tumor (ET) → Blue mesh
     - Handles both model output (label 3) and ground truth format (label 4)

3. **Mesh Generation:**
   - Extracts binary mask for each subregion
   - Applies marching cubes algorithm
   - Applies Laplacian smoothing
   - Exports as STL file

4. **Output Files:**
   - `necrotic_tumor.stl`
   - `edema_tumor.stl`
   - `enhancing_tumor.stl`

## Frontend Implementation

### Dependencies Added
- `three` - Three.js core library
- `@react-three/fiber` - React renderer for Three.js
- `@react-three/drei` - Useful helpers for Three.js

### Files Created/Modified

1. **`neurovision/src/components/tumor-3d-viewer.tsx`** (NEW)
   - `TumorMesh` - Component to load and display STL mesh
   - `Scene3D` - Main 3D scene with lighting and controls
   - `Tumor3DViewer` - Main component with visibility toggles

2. **`neurovision/src/components/FileUploadForm.tsx`** (MODIFIED)
   - Added import for `Tumor3DViewer`
   - Added 3D visualization section after results

3. **`neurovision/src/lib/types.ts`** (MODIFIED)
   - Added mesh URLs to `InferOutputs` type

### Features

- **Interactive Controls:**
  - Left-click + drag → Rotate
  - Scroll wheel → Zoom
  - Right-click + drag → Pan

- **Visibility Toggles:**
  - Checkboxes to show/hide each tumor subregion
  - Color-coded labels (Red, Green, Blue)

- **Color Scheme:**
  - Necrotic Tumor (NET) → Red (#ef4444)
  - Edema (ED) → Green (#22c55e)
  - Enhancing Tumor (ET) → Blue (#3b82f6)

## Installation

### Backend
```bash
cd server
pip install scikit-image trimesh
```

### Frontend
```bash
cd neurovision
npm install three @react-three/fiber @react-three/drei
```

## Usage

1. Upload ZIP file containing:
   - Required: `flair.nii`, `t1.nii`, `t1ce.nii`, `t2.nii`
   - Optional: `seg.nii` (ground truth segmentation)

2. Click "Start Inference"

3. After inference completes:
   - 3D meshes are automatically generated
   - "3D Tumor Visualization" section appears below results
   - Interactive viewer loads with all available meshes

4. Use controls:
   - Toggle visibility of each subregion
   - Rotate, zoom, and pan to explore the tumor

## Technical Details

### Mesh Generation
- Algorithm: Marching Cubes (from scikit-image)
- Smoothing: Laplacian smoothing applied
- Format: STL (binary)
- Voxel spacing: Extracted from NIfTI header (default: 1mm isotropic)

### 3D Rendering
- Framework: Three.js via React Three Fiber
- Material: MeshStandardMaterial with transparency
- Lighting: Ambient + 2 directional lights
- Camera: Perspective camera with orbit controls

## Notes

- Only tumor regions are rendered (no full brain volume)
- Meshes are generated on-demand after inference
- Empty masks (no tumor) result in no mesh file
- Mesh files are served with CORS headers for cross-origin loading
