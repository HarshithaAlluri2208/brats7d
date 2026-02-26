"use client";

import * as React from "react";
import type { InferResponse } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { ImageDisplay } from "@/components/image-display";

interface InferenceResultsProps {
  results: InferResponse;
}

export function InferenceResults({ results }: InferenceResultsProps) {
  const [overlayOpacity, setOverlayOpacity] = React.useState(50);
  const [showOverlays, setShowOverlays] = React.useState(true);

  const handleDownload = (url: string, filename: string) => {
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleOpenFullResolution = (url: string) => {
    window.open(url, "_blank", "noopener,noreferrer");
  };

  return (
    <div className="space-y-6 w-full">
      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">Whole Tumor</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-primary">
              {results.metrics.wt_volume_cc.toFixed(2)} cc
            </div>
            <p className="text-sm text-muted-foreground mt-1">
              Total tumor volume
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">Tumor Core</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-primary">
              {results.metrics.tc_volume_cc.toFixed(2)} cc
            </div>
            <p className="text-sm text-muted-foreground mt-1">
              Core tumor volume
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">Enhancing Tumor</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-primary">
              {results.metrics.et_volume_cc.toFixed(2)} cc
            </div>
            <p className="text-sm text-muted-foreground mt-1">
              Enhancing region volume
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Montages */}
      <Card>
        <CardHeader>
          <CardTitle>Montage Views</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Axial View</Label>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    handleOpenFullResolution(results.outputs.montage_axial)
                  }
                >
                  Open Full
                </Button>
              </div>
              <div className="relative rounded-lg border overflow-hidden bg-muted">
                <ImageDisplay
                  src={results.outputs.montage_axial}
                  alt="Axial montage"
                  className="w-full h-auto cursor-pointer hover:opacity-90 transition-opacity"
                  onLoad={() => {
                    console.log(`[ImageDisplay] Axial montage loaded: ${results.outputs.montage_axial}`);
                  }}
                  loading="lazy"
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Coronal View</Label>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    handleOpenFullResolution(results.outputs.montage_coronal)
                  }
                >
                  Open Full
                </Button>
              </div>
              <div className="relative rounded-lg border overflow-hidden bg-muted">
                <ImageDisplay
                  src={results.outputs.montage_coronal}
                  alt="Coronal montage"
                  className="w-full h-auto cursor-pointer hover:opacity-90 transition-opacity"
                  onLoad={() => {
                    console.log(`[ImageDisplay] Coronal montage loaded: ${results.outputs.montage_coronal}`);
                  }}
                  loading="lazy"
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Sagittal View</Label>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    handleOpenFullResolution(results.outputs.montage_sagittal)
                  }
                >
                  Open Full
                </Button>
              </div>
              <div className="relative rounded-lg border overflow-hidden bg-muted">
                <ImageDisplay
                  src={results.outputs.montage_sagittal}
                  alt="Sagittal montage"
                  className="w-full h-auto cursor-pointer hover:opacity-90 transition-opacity"
                  onLoad={() => {
                    console.log(`[ImageDisplay] Sagittal montage loaded: ${results.outputs.montage_sagittal}`);
                  }}
                  loading="lazy"
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Comparison Overlay (if available) */}
      {results.outputs.comparison_overlay && (
        <Card>
          <CardHeader>
            <CardTitle>Prediction vs Ground Truth Comparison</CardTitle>
            <p className="text-sm text-muted-foreground mt-2">
              Color-coded overlay: <span className="text-green-600 font-semibold">Green</span> = True Positive,{" "}
              <span className="text-yellow-600 font-semibold">Yellow</span> = False Positive,{" "}
              <span className="text-red-600 font-semibold">Red</span> = False Negative
            </p>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Comparison Overlay (Middle Axial Slice)</Label>
                <div className="flex gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleOpenFullResolution(results.outputs.comparison_overlay!)}
                  >
                    Open Full
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() =>
                      handleDownload(
                        results.outputs.comparison_overlay!,
                        `${results.job_id}_comparison_overlay.png`
                      )
                    }
                  >
                    Download
                  </Button>
                </div>
              </div>
              <div className="relative rounded-lg border overflow-hidden bg-muted">
                <ImageDisplay
                  src={results.outputs.comparison_overlay}
                  alt="Prediction vs Ground Truth Comparison"
                  className="w-full h-auto cursor-pointer hover:opacity-90 transition-opacity"
                  onLoad={() => {
                    console.log(`[ImageDisplay] Comparison overlay loaded: ${results.outputs.comparison_overlay}`);
                  }}
                  loading="lazy"
                />
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Class Overlays */}
      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <CardTitle>Class Overlays</CardTitle>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Switch
                  id="show-overlays"
                  checked={showOverlays}
                  onCheckedChange={setShowOverlays}
                />
                <Label htmlFor="show-overlays" className="cursor-pointer">
                  Show Overlays
                </Label>
              </div>
              {showOverlays && (
                <div className="flex items-center gap-3 min-w-[200px]">
                  <Label className="text-xs text-muted-foreground">
                    Opacity: {overlayOpacity}%
                  </Label>
                  <Slider
                    value={[overlayOpacity]}
                    onValueChange={(value) => setOverlayOpacity(value[0])}
                    min={0}
                    max={100}
                    step={5}
                    className="flex-1"
                  />
                </div>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              { id: 1, label: "Class 1 (NET)", url: results.outputs.overlay_class_1 },
              { id: 2, label: "Class 2 (ED)", url: results.outputs.overlay_class_2 },
              { id: 3, label: "Class 3 (ET)", url: results.outputs.overlay_class_3 },
            ].map((overlay) => (
              <div key={overlay.id} className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>{overlay.label}</Label>
                  <div className="flex gap-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleOpenFullResolution(overlay.url)}
                    >
                      Open
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        handleDownload(
                          overlay.url,
                          `overlay_class_${overlay.id}.png`
                        )
                      }
                    >
                      Download
                    </Button>
                  </div>
                </div>
                <div className="relative rounded-lg border overflow-hidden bg-muted">
                  <ImageDisplay
                    src={overlay.url}
                    alt={overlay.label}
                    className="w-full h-auto cursor-pointer hover:opacity-90 transition-opacity"
                    onLoad={() => {
                      console.log(`[ImageDisplay] ${overlay.label} overlay loaded: ${overlay.url}`);
                    }}
                    style={{
                      opacity: showOverlays ? overlayOpacity / 100 : 1,
                    }}
                    loading="lazy"
                  />
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Download Section */}
      <Card>
        <CardHeader>
          <CardTitle>Download Results</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {results.outputs.segmentation_nifti && (
              <Button
                variant="outline"
                className="w-full justify-start"
                onClick={() =>
                  handleDownload(
                    results.outputs.segmentation_nifti!,
                    `${results.job_id}_segmentation.nii.gz`
                  )
                }
              >
                <svg
                  className="mr-2 h-4 w-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                  />
                </svg>
                Download NIfTI
              </Button>
            )}

            <Button
              variant="outline"
              className="w-full justify-start"
              onClick={() =>
                handleDownload(
                  results.outputs.segmentation_numpy,
                  `${results.job_id}_segmentation.npy`
                )
              }
            >
              <svg
                className="mr-2 h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                />
              </svg>
              Download NumPy
            </Button>

            <Button
              variant="outline"
              className="w-full justify-start"
              onClick={() =>
                handleDownload(
                  results.outputs.probability_maps,
                  `${results.job_id}_probabilities.npz`
                )
              }
            >
              <svg
                className="mr-2 h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                />
              </svg>
              Download Probabilities
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

