"use client";

import * as React from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

interface Tumor3DViewerProps {
  meshUrls: {
    necrotic?: string | null;
    edema?: string | null;
    enhancing?: string | null;
    brain?: string | null;
  };
}

// Component to load and display a single STL mesh
function TumorMesh({
  url,
  color,
  visible,
  meshRef,
  onGeometryLoaded
}:
{
  url: string;
  color: string;
  visible: boolean;
  meshRef: React.RefObject<THREE.Mesh | null>;
  onGeometryLoaded?: () => void;
}) {
  const [geometry, setGeometry] = React.useState<THREE.BufferGeometry | null>(null);

  React.useEffect(() => {
    const loader = new STLLoader();
    loader.load(
      url,
      (loadedGeometry) => {
        setGeometry(loadedGeometry);
        onGeometryLoaded?.();
      },
      undefined,
      (error) => {
        console.error(`Failed to load STL mesh from ${url}:`, error);
      }
    );
  }, [url, onGeometryLoaded]);

  React.useEffect(() => {
    if (meshRef.current) {
      meshRef.current.visible = visible;
    }
  }, [visible, meshRef]);

  if (!geometry) return null;

  return (
    <mesh
      ref={meshRef}
      geometry={geometry}
      visible={visible}
      userData={{ isTumorMesh: true }}>
      <meshStandardMaterial
        color={color}
        metalness={0.1}
        roughness={0.7}
        transparent={true}
        opacity={visible ? 0.9 : 0}
      />
    </mesh>
  );
}

// Component to load and display brain mesh
function BrainMesh({
  url,
  visible,
  meshRef,
  onLoaded
}:
{
  url: string;
  visible: boolean;
  meshRef: React.RefObject<THREE.Mesh | null>;
  onLoaded?: (center: THREE.Vector3) => void;
}) {
  const [geometry, setGeometry] = React.useState<THREE.BufferGeometry | null>(null);

  React.useEffect(() => {
    const loader = new STLLoader();
    loader.load(
      url,
      (loadedGeometry) => {
        setGeometry(loadedGeometry);
        // Compute bounding box center
        loadedGeometry.computeBoundingBox();
        const box = loadedGeometry.boundingBox;
        if (box && onLoaded) {
          const center = new THREE.Vector3();
          box.getCenter(center);
          onLoaded(center);
        }
      },
      undefined,
      (error) => {
        console.error(`Failed to load brain STL mesh from ${url}:`, error);
      }
    );
  }, [url, onLoaded]);

  React.useEffect(() => {
    if (meshRef.current) {
      meshRef.current.visible = visible;
    }
  }, [visible, meshRef]);

  if (!geometry) return null;

  return (
    <mesh
      ref={meshRef}
      geometry={geometry}
      visible={visible}
      userData={{ isBrainMesh: true }}>
      <meshStandardMaterial
        color="#d1d5db"
        metalness={0.0}
        roughness={1.0}
        transparent={true}
        opacity={0.2}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

// Click handler component using Raycaster
function ClickHandler({
  brainCenter,
  onPointSelected
}: {
  brainCenter: React.MutableRefObject<THREE.Vector3>;
  onPointSelected: (info: { hemisphere: string; direction: string; depth: string }) => void;
}) {
  const { camera, scene, gl } = useThree();
  const raycaster = React.useRef(new THREE.Raycaster());
  const mouse = React.useRef(new THREE.Vector2());
  const mouseDownPos = React.useRef<{ x: number; y: number } | null>(null);

  React.useEffect(() => {
    const handleMouseDown = (event: MouseEvent) => {
      // Store initial mouse position
      mouseDownPos.current = { x: event.clientX, y: event.clientY };
    };

    const handleClick = (event: MouseEvent) => {
      // Only process if mouse didn't move much (actual click, not drag)
      if (mouseDownPos.current) {
        const dx = Math.abs(event.clientX - mouseDownPos.current.x);
        const dy = Math.abs(event.clientY - mouseDownPos.current.y);
        
        // If mouse moved more than 5 pixels, it was a drag, not a click
        if (dx > 5 || dy > 5) {
          mouseDownPos.current = null;
          return;
        }
        mouseDownPos.current = null;
      }

      // Get mouse position in normalized device coordinates
      const rect = gl.domElement.getBoundingClientRect();
      mouse.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      // Update raycaster
      raycaster.current.setFromCamera(mouse.current, camera);

      // Find intersections with all objects in scene
      const intersects = raycaster.current.intersectObjects(scene.children, true);

      // Filter to only tumor meshes (check userData)
      const tumorIntersects = intersects.filter(
        (intersect) => intersect.object.userData.isTumorMesh === true
      );

      if (tumorIntersects.length > 0) {
        const intersection = tumorIntersects[0];
        const point = intersection.point;

        // Determine spatial orientation relative to brain center
        const hemisphere = point.x < brainCenter.current.x ? "Left" : "Right";
        const direction = point.y < brainCenter.current.y ? "Posterior" : "Anterior";
        const depth = point.z < brainCenter.current.z ? "Inferior" : "Superior";

        onPointSelected({ hemisphere, direction, depth });
      }
    };

    gl.domElement.addEventListener("mousedown", handleMouseDown);
    gl.domElement.addEventListener("click", handleClick);
    return () => {
      gl.domElement.removeEventListener("mousedown", handleMouseDown);
      gl.domElement.removeEventListener("click", handleClick);
    };
  }, [camera, scene, gl, brainCenter, onPointSelected]);

  return null;
}

// Main 3D scene component
function Scene3D({
  meshUrls,
  visibility,
  onPointSelected
}: {
  meshUrls: Tumor3DViewerProps["meshUrls"];
  visibility: { [key: string]: boolean };
  onPointSelected: (info: { hemisphere: string; direction: string; depth: string }) => void;
}) {
  const brainCenter = React.useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 0));
  const necroticMeshRef = React.useRef<THREE.Mesh>(null);
  const edemaMeshRef = React.useRef<THREE.Mesh>(null);
  const enhancingMeshRef = React.useRef<THREE.Mesh>(null);
  const brainMeshRef = React.useRef<THREE.Mesh>(null);
  const brainCenterComputed = React.useRef(false);

  const handleBrainLoaded = React.useCallback((center: THREE.Vector3) => {
    brainCenter.current.copy(center);
    brainCenterComputed.current = true;
  }, []);

  // Function to compute center from tumor meshes
  const computeCenterFromTumors = React.useCallback(() => {
    if (!meshUrls.brain && !brainCenterComputed.current) {
      const meshes = [necroticMeshRef.current, edemaMeshRef.current, enhancingMeshRef.current].filter(Boolean) as THREE.Mesh[];
      if (meshes.length > 0) {
        const box = new THREE.Box3();
        let hasValidGeometry = false;
        
        meshes.forEach(mesh => {
          if (mesh.geometry) {
            mesh.geometry.computeBoundingBox();
            if (mesh.geometry.boundingBox) {
              const meshBox = mesh.geometry.boundingBox;
              box.expandByPoint(meshBox.min);
              box.expandByPoint(meshBox.max);
              hasValidGeometry = true;
            }
          }
        });
        
        if (hasValidGeometry && !box.isEmpty()) {
          const center = new THREE.Vector3();
          box.getCenter(center);
          brainCenter.current.copy(center);
          brainCenterComputed.current = true;
        }
      }
    }
  }, [meshUrls.brain]);

  // Compute center when meshes load
  React.useEffect(() => {
    computeCenterFromTumors();
  }, [computeCenterFromTumors, visibility.necrotic, visibility.edema, visibility.enhancing]);

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <directionalLight position={[-10, -10, -5]} intensity={0.5} />

      {/* Camera */}
      <PerspectiveCamera makeDefault position={[200, 200, 200]} fov={50} />

      {/* Load and display brain mesh first */}
      {meshUrls.brain && (
        <BrainMesh
          url={meshUrls.brain}
          visible={visibility.brain ?? true}
          meshRef={brainMeshRef}
          onLoaded={handleBrainLoaded}
        />
      )}

      {/* Load and display tumor meshes */}
      {meshUrls.necrotic && (
        <TumorMesh
          url={meshUrls.necrotic}
          color="#ef4444" // Red
          visible={visibility.necrotic}
          meshRef={necroticMeshRef}
          onGeometryLoaded={computeCenterFromTumors}
        />
      )}
      {meshUrls.edema && (
        <TumorMesh
          url={meshUrls.edema}
          color="#22c55e" // Green
          visible={visibility.edema}
          meshRef={edemaMeshRef}
          onGeometryLoaded={computeCenterFromTumors}
        />
      )}
      {meshUrls.enhancing && (
        <TumorMesh
          url={meshUrls.enhancing}
          color="#3b82f6" // Blue
          visible={visibility.enhancing}
          meshRef={enhancingMeshRef}
          onGeometryLoaded={computeCenterFromTumors}
        />
      )}

      {/* Click handler */}
      <ClickHandler
        brainCenter={brainCenter}
        onPointSelected={onPointSelected}
      />

      {/* Controls */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={50}
        maxDistance={500}
      />
    </>
  );
}

export function Tumor3DViewer({ meshUrls }: Tumor3DViewerProps) {
  const [visibility, setVisibility] = React.useState({
    necrotic: true,
    edema: true,
    enhancing: true,
  });

  const [locationInfo, setLocationInfo] = React.useState<{
    hemisphere: string;
    direction: string;
    depth: string;
  } | null>(null);
  
  

  const hasAnyMesh = meshUrls.necrotic || meshUrls.edema || meshUrls.enhancing;

  if (!hasAnyMesh) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>3D Tumor Visualization</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No 3D meshes available. Make sure seg.nii is included in your upload.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>3D Tumor Visualization</CardTitle>
        <p className="text-sm text-muted-foreground mt-2">
          Interactive 3D view of tumor subregions. Rotate with mouse, zoom with scroll wheel.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Visibility Controls */}
        <div className="space-y-3 p-3 border rounded-md bg-muted/50">
          <Label className="text-sm font-semibold">Toggle Visibility</Label>
          <div className="space-y-2">
            {meshUrls.necrotic && (
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
            {meshUrls.edema && (
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
            {meshUrls.enhancing && (
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
          </div>
        </div>

        {/* 3D Canvas */}
        <div className="w-full h-[500px] border rounded-lg overflow-hidden bg-black relative">
          <Canvas>
            <Scene3D meshUrls={meshUrls} visibility={visibility} onPointSelected={setLocationInfo}/>
          </Canvas>
          
          {/* Floating information panel */}
          {locationInfo && (
            <div className="absolute bottom-4 right-4 p-4 border rounded-lg bg-gray-900/95 text-white text-sm shadow-lg z-10 min-w-[200px]">
              <div className="font-semibold mb-2 text-base">Tumour Spatial Location</div>
              <div className="space-y-1">
                <div>Hemisphere: <span className="font-medium">{locationInfo.hemisphere}</span></div>
                <div>Direction: <span className="font-medium">{locationInfo.direction}</span></div>
                <div>Depth: <span className="font-medium">{locationInfo.depth}</span></div>
              </div>
            </div>
          )}
        </div>

        <p className="text-xs text-muted-foreground">
          <strong>Controls:</strong> Left-click + drag to rotate • Scroll to zoom • Right-click + drag to pan
        </p>
      </CardContent>
    </Card>
  );
}
