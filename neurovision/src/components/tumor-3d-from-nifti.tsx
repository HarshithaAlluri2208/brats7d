"use client";

import * as React from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera, Bounds } from "@react-three/drei";
import * as THREE from "three";
import { readHeader, readImage, isCompressed, decompress } from "nifti-reader-js";
import type { NIFTI1, NIFTI2 } from "nifti-reader-js";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

interface Tumor3DFromNiftiProps {
  segNiftiUrl: string;
  mriUrl?: string; // Optional MRI URL for brain mask (t1 or t1ce)
}

// Simple marching cubes implementation for extracting surfaces
function marchingCubes(
  volume: Uint8Array,
  dims: [number, number, number],
  threshold: number
): Float32Array {
  const vertices: number[] = [];
  const [nx, ny, nz] = dims;

  // Simplified marching cubes - creates a basic surface
  for (let z = 0; z < nz - 1; z++) {
    for (let y = 0; y < ny - 1; y++) {
      for (let x = 0; x < nx - 1; x++) {
        const idx = x + y * nx + z * nx * ny;
        const val = volume[idx];

        if (val === threshold) {
          // Add vertices for this voxel (simplified - just create a cube)
          const px = x - nx / 2;
          const py = y - ny / 2;
          const pz = z - nz / 2;

          // Create 8 vertices of a cube
          const cubeVertices = [
            [px, py, pz],
            [px + 1, py, pz],
            [px + 1, py + 1, pz],
            [px, py + 1, pz],
            [px, py, pz + 1],
            [px + 1, py, pz + 1],
            [px + 1, py + 1, pz + 1],
            [px, py + 1, pz + 1],
          ];

          // Add faces (simplified - just add all vertices)
          for (const v of cubeVertices) {
            vertices.push(...v);
          }
        }
      }
    }
  }

  return new Float32Array(vertices);
}

// Extract brain mask from segmentation (all non-zero voxels = brain tissue)
function extractBrainMask(
  volume: Uint8Array,
  dims: [number, number, number]
): Uint8Array {
  const brainMask = new Uint8Array(volume.length);
  for (let i = 0; i < volume.length; i++) {
    // Any non-zero value is brain tissue
    brainMask[i] = volume[i] > 0 ? 1 : 0;
  }
  return brainMask;
}

// Extract surface vertices using a simplified approach
// Creates vertices only for surface voxels to reduce geometry complexity
function extractSurface(
  volume: Uint8Array,
  dims: [number, number, number],
  label: number
): Float32Array {
  const vertices: number[] = [];
  const [nx, ny, nz] = dims;
  const voxelSize = 1.0; // Adjust based on your NIfTI spacing if needed

  // Extract only surface voxels with the specific label
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const idx = x + y * nx + z * nx * ny;
        const val = volume[idx];

        if (val === label) {
          // Check if this voxel is on the surface (has a neighbor with different value)
          let isSurface = false;
          const neighbors = [
            [x + 1, y, z],
            [x - 1, y, z],
            [x, y + 1, z],
            [x, y - 1, z],
            [x, y, z + 1],
            [x, y, z - 1],
          ];

          for (const [nxVal, nyVal, nzVal] of neighbors) {
            if (
              nxVal >= 0 &&
              nxVal < dims[0] &&
              nyVal >= 0 &&
              nyVal < dims[1] &&
              nzVal >= 0 &&
              nzVal < dims[2]
            ) {
              const nIdx = nxVal + nyVal * dims[0] + nzVal * dims[0] * dims[1];
              if (volume[nIdx] !== label) {
                isSurface = true;
                break;
              }
            }
          }

          if (isSurface) {
            // Convert to world coordinates (center the volume)
            const px = (x - nx / 2) * voxelSize;
            const py = (y - ny / 2) * voxelSize;
            const pz = (z - nz / 2) * voxelSize;

            // Create vertices for a cube representing this voxel
            // Using a smaller size to avoid gaps
            const size = voxelSize * 0.9;
            const halfSize = size / 2;

            // Define 8 vertices of a cube
            const cubeVertices = [
              [px - halfSize, py - halfSize, pz - halfSize],
              [px + halfSize, py - halfSize, pz - halfSize],
              [px + halfSize, py + halfSize, pz - halfSize],
              [px - halfSize, py + halfSize, pz - halfSize],
              [px - halfSize, py - halfSize, pz + halfSize],
              [px + halfSize, py - halfSize, pz + halfSize],
              [px + halfSize, py + halfSize, pz + halfSize],
              [px - halfSize, py + halfSize, pz + halfSize],
            ];

            // Add all vertices (will be used to create faces)
            for (const [vx, vy, vz] of cubeVertices) {
              vertices.push(vx, vy, vz);
            }
          }
        }
      }
    }
  }

  return new Float32Array(vertices);
}

// Component to render a single tumor subregion mesh
function TumorSubregionMesh({
  vertices,
  color,
  visible,
  locationMode,
  onSelectPoint,
}: {
  vertices: Float32Array;
  color: string;
  visible: boolean;
  locationMode: boolean;
  onSelectPoint?: (point: THREE.Vector3) => void;
}) {
  const geometry = React.useMemo(() => {
    if (vertices.length === 0) return null;

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
    geom.computeVertexNormals();
    return geom;
  }, [vertices]);

  if (!geometry || vertices.length === 0 || !visible) return null;

  return (
    <mesh
      geometry={geometry}
      onClick={(e) => {
        if (!locationMode) return;
        if (e?.point && onSelectPoint) {
          onSelectPoint(e.point.clone());
        }
      }}
    >
      <meshStandardMaterial
        color={color}
        transparent
        opacity={0.7}
        metalness={0.1}
        roughness={0.7}
      />
    </mesh>
  );
}


// Component to render brain mesh
function BrainMesh({
  vertices,
  visible,
}: {
  vertices: Float32Array;
  visible: boolean;
}) {
  const geometry = React.useMemo(() => {
    if (vertices.length === 0) return null;
    
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
    geom.computeVertexNormals();
    return geom;
  }, [vertices]);

  if (!geometry || vertices.length === 0 || !visible) return null;

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial
        color="#d1d5db" // Light gray
        transparent={true}
        opacity={0.2} // Low opacity (0.15-0.25 range)
        metalness={0.0}
        roughness={1.0} // Smooth, non-reflective
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

// Main 3D scene component
function Scene3D({
  meshes,
  brainMesh,
  visibility,
  locationMode,
  onSelectPoint,
}: {
  meshes: {
    necrotic: Float32Array;
    edema: Float32Array;
    enhancing: Float32Array;
  };
  brainMesh: Float32Array;
  visibility: {
    necrotic: boolean;
    edema: boolean;
    enhancing: boolean;
    brain: boolean;
  };
  locationMode: boolean;
  onSelectPoint?: (point: THREE.Vector3) => void;
}) {
  const groupRef = React.useRef<THREE.Group>(null);

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <directionalLight position={[-10, -10, -5]} intensity={0.5} />

      {/* Camera */}
      <PerspectiveCamera makeDefault position={[150, 150, 150]} fov={50} />

      {/* Brain and Tumor meshes */}
      <Bounds fit clip observe margin={1.5}>
        <group ref={groupRef}>
          {/* Brain mesh - rendered first (behind tumors) */}
          {brainMesh.length > 0 && (
            <BrainMesh
              vertices={brainMesh}
              visible={visibility.brain}
            />
          )}
          
          {/* Tumor meshes - rendered on top (visually dominant) */}
          {meshes.necrotic.length > 0 && (
            <TumorSubregionMesh
              vertices={meshes.necrotic}
              color="#ef4444" // Red
              visible={visibility.necrotic}
              locationMode={locationMode}
              onSelectPoint={onSelectPoint}
            />
          )}
          {meshes.edema.length > 0 && (
            <TumorSubregionMesh
              vertices={meshes.edema}
              color="#22c55e" // Green
              visible={visibility.edema}
              locationMode={locationMode}
              onSelectPoint={onSelectPoint}
            />
          )}
          {meshes.enhancing.length > 0 && (
            <TumorSubregionMesh
              vertices={meshes.enhancing}
              color="#3b82f6" // Blue
              visible={visibility.enhancing}
              locationMode={locationMode}
              onSelectPoint={onSelectPoint}
            />
          )}
        </group>
      </Bounds>

      {/* Controls */}
      <OrbitControls
        enabled={!locationMode}
        enablePan={!locationMode}
        enableZoom={!locationMode}
        enableRotate={!locationMode}
        minDistance={50}
        maxDistance={500}
      />
    </>
  );
}

export function Tumor3DFromNifti({ segNiftiUrl, mriUrl }: Tumor3DFromNiftiProps) {
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [meshes, setMeshes] = React.useState<{
    necrotic: Float32Array;
    edema: Float32Array;
    enhancing: Float32Array;
    brain: Float32Array;
  }>({
    necrotic: new Float32Array(0),
    edema: new Float32Array(0),
    enhancing: new Float32Array(0),
    brain: new Float32Array(0),
  });
  const [visibility, setVisibility] = React.useState({
    necrotic: true,
    edema: true,
    enhancing: true,
    brain: true,
  });
  const [locationMode, setLocationMode] = React.useState(false);

  // New hover state: store spatial orientation derived from hovered world point
  const [locationInfo, setLocationInfo] = React.useState<{
    hemisphere: string;
    direction: string;
    depth: string;
  } | null>(null);

  React.useEffect(() => {
    async function loadAndProcessNifti() {
      try {
        setLoading(true);
        setError(null);

        // Fetch the NIfTI file
        const response = await fetch(segNiftiUrl);
        if (!response.ok) {
          throw new Error(`Failed to fetch NIfTI file: ${response.statusText}`);
        }

        let arrayBuffer = await response.arrayBuffer();

        // Decompress if needed
        if (isCompressed(arrayBuffer)) {
          arrayBuffer = decompress(arrayBuffer) as ArrayBuffer;
        }

        // Read header
        const header = readHeader(arrayBuffer);
        if (!header) {
          throw new Error("Failed to read NIfTI header");
        }

        // Get dimensions
        const dims: [number, number, number] = [
          header.dims[1],
          header.dims[2],
          header.dims[3],
        ];

        // Read image data
        const imageBuffer = readImage(header, arrayBuffer);
        if (!imageBuffer) {
          throw new Error("Failed to read NIfTI image data");
        }

        // Convert image data to Uint8Array based on datatype
        let volumeData: Uint8Array;
        const numVoxels = dims[0] * dims[1] * dims[2];

        // Handle different data types
        // TYPE_UINT8 = 2, TYPE_INT16 = 4, TYPE_FLOAT32 = 16
        if (header.datatypeCode === 2) {
          // UINT8
          volumeData = new Uint8Array(imageBuffer);
        } else if (header.datatypeCode === 4) {
          // INT16
          const int16Data = new Int16Array(imageBuffer);
          volumeData = new Uint8Array(numVoxels);
          for (let i = 0; i < numVoxels; i++) {
            volumeData[i] = Math.max(0, Math.min(255, int16Data[i]));
          }
        } else if (header.datatypeCode === 16) {
          // FLOAT32
          const float32Data = new Float32Array(imageBuffer);
          volumeData = new Uint8Array(numVoxels);
          for (let i = 0; i < numVoxels; i++) {
            volumeData[i] = Math.round(Math.max(0, Math.min(255, float32Data[i])));
          }
        } else {
          // Default: try to convert to Uint8Array
          volumeData = new Uint8Array(imageBuffer);
        }

        // Extract surfaces for each label
        // Label 1 → Necrotic Tumor (NET)
        // Label 2 → Edema (ED)
        // Label 4 → Enhancing Tumor (ET)
        const necroticVertices = extractSurface(volumeData, dims, 1);
        const edemaVertices = extractSurface(volumeData, dims, 2);
        const enhancingVertices = extractSurface(volumeData, dims, 4);

        // Generate brain mask from segmentation (all non-zero voxels = brain tissue)
        const brainMask = extractBrainMask(volumeData, dims);
        const brainVertices = extractSurface(brainMask, dims, 1);

        setMeshes({
          necrotic: necroticVertices,
          edema: edemaVertices,
          enhancing: enhancingVertices,
          brain: brainVertices,
        });

        setLoading(false);
      } catch (err) {
        console.error("Error loading NIfTI:", err);
        setError(
          err instanceof Error ? err.message : "Failed to load NIfTI file"
        );
        setLoading(false);
      }
    }

    if (segNiftiUrl) {
      loadAndProcessNifti();
    }
  }, [segNiftiUrl]);

  const hasAnyMesh =
    meshes.necrotic.length > 0 ||
    meshes.edema.length > 0 ||
    meshes.enhancing.length > 0 ||
    meshes.brain.length > 0;

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>3D Tumor Visualization</CardTitle>
        </CardHeader>
        <CardContent className="py-10 text-center">
          <p className="text-muted-foreground">Loading and processing segmentation...</p>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>3D Tumor Visualization</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-destructive">{error}</p>
        </CardContent>
      </Card>
    );
  }

  if (!hasAnyMesh) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>3D Tumor Visualization</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No tumor subregions found in segmentation. Make sure the segmentation
            includes labels 1 (NET), 2 (ED), and 4 (ET).
          </p>
        </CardContent>
      </Card>
    );
  }

  // Handler to compute orientation relative to origin (volume centered at origin)
  const handleSelectPoint = (point: THREE.Vector3) => {
    const hemisphere =
      point.x < 0 ? "Left" : point.x > 0 ? "Right" : "Midline";
    const direction =
      point.y < 0 ? "Posterior" : point.y > 0 ? "Anterior" : "Midline";
    const depth =
      point.z < 0 ? "Inferior" : point.z > 0 ? "Superior" : "Midline";

    setLocationInfo({ hemisphere, direction, depth });
  };


  return (
    <Card>
      <CardHeader>
        <CardTitle>3D Tumor Visualization</CardTitle>
        <p className="text-sm text-muted-foreground mt-2">
          Interactive 3D view of tumor subregions extracted from segmentation.
          Rotate with mouse, zoom with scroll wheel.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Legend */}
        <div className="p-3 border rounded-md bg-muted/30">
          <Label className="text-sm font-semibold mb-2 block">Color Legend</Label>
          <div className="space-y-1 text-xs">
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded bg-red-500" />
              <span>Label 1: Necrotic Tumor (NET)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded bg-green-500" />
              <span>Label 2: Edema (ED)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded bg-blue-500" />
              <span>Label 4: Enhancing Tumor (ET)</span>
            </div>
          </div>
        </div>

        {/* Visibility Controls */}
        <div className="space-y-3 p-3 border rounded-md bg-muted/50">
          <Label className="text-sm font-semibold">Toggle Visibility</Label>
          <div className="space-y-2">
            {meshes.necrotic.length > 0 && (
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="h-4 w-4 rounded bg-red-500" />
                  <Label htmlFor="necrotic" className="cursor-pointer">
                    Necrotic Tumor (NET)
                  </Label>
                </div>
                <Switch
                  id="necrotic"
                  checked={visibility.necrotic}
                  onCheckedChange={(checked) =>
                    setVisibility((prev) => ({ ...prev, necrotic: checked }))
                  }
                />
              </div>
            )}
            {meshes.edema.length > 0 && (
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="h-4 w-4 rounded bg-green-500" />
                  <Label htmlFor="edema" className="cursor-pointer">
                    Edema (ED)
                  </Label>
                </div>
                <Switch
                  id="edema"
                  checked={visibility.edema}
                  onCheckedChange={(checked) =>
                    setVisibility((prev) => ({ ...prev, edema: checked }))
                  }
                />
              </div>
            )}
            {meshes.enhancing.length > 0 && (
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="h-4 w-4 rounded bg-blue-500" />
                  <Label htmlFor="enhancing" className="cursor-pointer">
                    Enhancing Tumor (ET)
                  </Label>
                </div>
                <Switch
                  id="enhancing"
                  checked={visibility.enhancing}
                  onCheckedChange={(checked) =>
                    setVisibility((prev) => ({ ...prev, enhancing: checked }))
                  }
                />
              </div>
            )}
            {meshes.brain.length > 0 && (
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="h-4 w-4 rounded bg-gray-400" />
                  <Label htmlFor="brain" className="cursor-pointer">
                    Brain Structure
                  </Label>
                </div>
                <Switch
                  id="brain"
                  checked={visibility.brain}
                  onCheckedChange={(checked) =>
                    setVisibility((prev) => ({ ...prev, brain: checked }))
                  }
                />
              </div>
            )}
          </div>
        </div>
        
        {/* Location Detection Mode Toggle */}
        <div className="flex items-center justify-between p-3 border rounded-md bg-muted/50">
          <Label className="text-sm font-semibold">
            Location Detection Mode
          </Label>
          <Switch
            checked={locationMode}
            onCheckedChange={(checked) => {
              setLocationMode(checked);
              setLocationInfo(null); // clear old result when toggling
            }}
          />
        </div>

        {/* 3D Canvas */}
        <div className="w-full h-[600px] border rounded-lg overflow-hidden bg-black">
          <Canvas>
            <Scene3D 
              meshes={meshes} 
              brainMesh={meshes.brain} 
              visibility={visibility} 
              locationMode={locationMode}
              onSelectPoint={handleSelectPoint}
            />
          </Canvas>
        </div>

        {/* Hover info panel (under the canvas) */}
        {locationInfo && (
          <div className="p-3 mt-2 border rounded-md bg-gray-900 text-white text-sm shadow">
            <div className="font-semibold mb-1">Tumour Spatial Location</div>
            <div>Hemisphere: <span className="font-medium">{locationInfo.hemisphere}</span></div>
            <div>Direction: <span className="font-medium">{locationInfo.direction}</span></div>
            <div>Depth: <span className="font-medium">{locationInfo.depth}</span></div>
          </div>
        )}


        {/*<div style={{ color: "yellow", fontWeight: "bold", padding: "6px" }}>
          TEST — 3D FILE ACTIVE
        </div>*/}


        <p className="text-xs text-muted-foreground">
          <strong>Controls:</strong> Left-click + drag to rotate • Scroll to zoom • Right-click + drag to pan
        </p>
      </CardContent>
    </Card>
  );
}
