"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";

interface SliceViewerProps {
  // Base volume image URLs (one per slice) or single montage URL
  baseImageUrls?: string[];
  // Overlay image URLs per class, per slice: { class_1: string[], class_2: string[], class_3: string[] }
  overlayUrls?: {
    class_1?: string[];
    class_2?: string[];
    class_3?: string[];
  };
  // Or single overlay URLs if server provides per-slice images
  overlayImageUrls?: {
    class_1?: string[];
    class_2?: string[];
    class_3?: string[];
  };
  // Initial view orientation
  initialView?: "axial" | "coronal" | "sagittal";
  // Initial slice index
  initialSlice?: number;
}

type ViewOrientation = "axial" | "coronal" | "sagittal";

export function SliceViewer({
  baseImageUrls = [],
  overlayUrls,
  overlayImageUrls,
  initialView = "axial",
  initialSlice = 0,
}: SliceViewerProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [view, setView] = React.useState<ViewOrientation>(initialView);
  const [sliceIndex, setSliceIndex] = React.useState(initialSlice);
  const [classOpacity, setClassOpacity] = React.useState({
    class_1: 50,
    class_2: 50,
    class_3: 50,
  });
  const [classEnabled, setClassEnabled] = React.useState({
    class_1: true,
    class_2: true,
    class_3: true,
  });
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  // Use overlayImageUrls if provided, otherwise overlayUrls
  const overlays = overlayImageUrls || overlayUrls || {};

  // Get current slice count based on view
  const getSliceCount = () => {
    if (baseImageUrls.length > 0) return baseImageUrls.length;
    // If no base images, use overlay length as reference
    const overlayKeys = Object.keys(overlays);
    if (overlayKeys.length > 0) {
      const firstOverlay = overlays[overlayKeys[0] as keyof typeof overlays];
      if (Array.isArray(firstOverlay)) return firstOverlay.length;
    }
    return 1;
  };

  const sliceCount = getSliceCount();
  const maxSlice = Math.max(0, sliceCount - 1);

  // Load and render images on canvas
  const renderCanvas = React.useCallback(async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    setIsLoading(true);
    setError(null);

    try {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Load base image
      let baseImage: HTMLImageElement | null = null;
      if (baseImageUrls.length > sliceIndex) {
        baseImage = await loadImage(baseImageUrls[sliceIndex]);
      }

      // Set canvas size
      if (baseImage) {
        canvas.width = baseImage.width;
        canvas.height = baseImage.height;
      } else {
        // Default size if no base image
        canvas.width = 512;
        canvas.height = 512;
      }

      // Draw base image
      if (baseImage) {
        ctx.drawImage(baseImage, 0, 0);
      } else {
        // Draw placeholder if no base image
        ctx.fillStyle = "#1a1a1a";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#666";
        ctx.font = "20px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("No base image available", canvas.width / 2, canvas.height / 2);
      }

      // Draw overlays
      const overlayClasses = [
        { key: "class_1", color: "rgba(255, 0, 0" }, // Red
        { key: "class_2", color: "rgba(0, 255, 0" }, // Green
        { key: "class_3", color: "rgba(0, 0, 255" }, // Blue
      ] as const;

      for (const overlayClass of overlayClasses) {
        if (!classEnabled[overlayClass.key as keyof typeof classEnabled]) continue;

        const overlayArray = overlays[overlayClass.key as keyof typeof overlays];
        if (!overlayArray || !Array.isArray(overlayArray)) continue;
        if (overlayArray.length <= sliceIndex) continue;

        const overlayUrl = overlayArray[sliceIndex];
        if (!overlayUrl) continue;

        try {
          const overlayImage = await loadImage(overlayUrl);
          const opacity = classOpacity[overlayClass.key as keyof typeof classOpacity] / 100;

          // Set global alpha for overlay
          ctx.globalAlpha = opacity;
          ctx.drawImage(overlayImage, 0, 0, canvas.width, canvas.height);

          // Reset global alpha
          ctx.globalAlpha = 1.0;
        } catch (err) {
          console.warn(`Failed to load overlay ${overlayClass.key}:`, err);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to render canvas");
      console.error("Render error:", err);
    } finally {
      setIsLoading(false);
    }
  }, [
    baseImageUrls,
    overlays,
    sliceIndex,
    classOpacity,
    classEnabled,
  ]);

  // Load image helper
  const loadImage = (url: string): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url;
    });
  };

  // Re-render when dependencies change
  React.useEffect(() => {
    renderCanvas();
  }, [renderCanvas]);

  // Handle slice change
  const handleSliceChange = (value: number[]) => {
    const newIndex = Math.max(0, Math.min(maxSlice, value[0]));
    setSliceIndex(newIndex);
  };

  // Handle class opacity change
  const handleClassOpacityChange = (classKey: keyof typeof classOpacity, value: number[]) => {
    setClassOpacity((prev) => ({
      ...prev,
      [classKey]: value[0],
    }));
  };

  // Handle class toggle
  const handleClassToggle = (classKey: keyof typeof classEnabled) => {
    setClassEnabled((prev) => ({
      ...prev,
      [classKey]: !prev[classKey],
    }));
  };

  // Download current view as PNG
  const handleDownload = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.toBlob((blob) => {
      if (!blob) return;

      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `slice_${view}_${sliceIndex}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }, "image/png");
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <CardTitle>Slice Viewer</CardTitle>
          <div className="flex items-center gap-2">
            <Button
              variant={view === "axial" ? "default" : "outline"}
              size="sm"
              onClick={() => setView("axial")}
            >
              Axial
            </Button>
            <Button
              variant={view === "coronal" ? "default" : "outline"}
              size="sm"
              onClick={() => setView("coronal")}
            >
              Coronal
            </Button>
            <Button
              variant={view === "sagittal" ? "default" : "outline"}
              size="sm"
              onClick={() => setView("sagittal")}
            >
              Sagittal
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Slice Navigation */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label>
              Slice: {sliceIndex + 1} / {sliceCount}
            </Label>
            <Button variant="outline" size="sm" onClick={handleDownload}>
              Download PNG
            </Button>
          </div>
          <Slider
            value={[sliceIndex]}
            onValueChange={handleSliceChange}
            min={0}
            max={maxSlice}
            step={1}
            className="w-full"
          />
        </div>

        {/* Canvas */}
        <div className="relative w-full flex justify-center bg-muted rounded-lg border overflow-hidden">
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/50 z-10">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            </div>
          )}
          {error && (
            <div className="absolute inset-0 flex items-center justify-center bg-destructive/10 z-10">
              <p className="text-sm text-destructive">{error}</p>
            </div>
          )}
          <canvas
            ref={canvasRef}
            className="max-w-full h-auto"
            style={{ maxHeight: "600px" }}
          />
        </div>

        {/* Overlay Controls */}
        <div className="space-y-3">
          <Label className="text-sm font-semibold">Overlay Controls</Label>
          {[
            { key: "class_1", label: "Class 1 (NET)", color: "text-red-500" },
            { key: "class_2", label: "Class 2 (ED)", color: "text-green-500" },
            { key: "class_3", label: "Class 3 (ET)", color: "text-blue-500" },
          ].map(({ key, label, color }) => {
            const classKey = key as keyof typeof classEnabled;
            const hasOverlay = overlays[classKey] && Array.isArray(overlays[classKey]);

            if (!hasOverlay) return null;

            return (
              <div key={key} className="space-y-2 p-3 rounded-lg border">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Switch
                      id={`toggle-${key}`}
                      checked={classEnabled[classKey]}
                      onCheckedChange={() => handleClassToggle(classKey)}
                    />
                    <Label
                      htmlFor={`toggle-${key}`}
                      className={`cursor-pointer ${color} font-medium`}
                    >
                      {label}
                    </Label>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {classOpacity[classKey]}%
                  </span>
                </div>
                {classEnabled[classKey] && (
                  <Slider
                    value={[classOpacity[classKey]]}
                    onValueChange={(value) =>
                      handleClassOpacityChange(classKey, value)
                    }
                    min={0}
                    max={100}
                    step={5}
                    className="w-full"
                    disabled={!classEnabled[classKey]}
                  />
                )}
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

