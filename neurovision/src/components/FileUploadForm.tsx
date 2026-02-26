"use client";

import * as React from "react";
import { inferWithFiles, inferWithZip } from "@/lib/api";
import type { InferFilesInput, InferResponse } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ImageDisplay } from "@/components/image-display";
import { Tumor3DViewer } from "@/components/tumor-3d-viewer";
import JSZip from "jszip";
import { useRouter } from "next/navigation";

export function FileUploadForm() {
  const router = useRouter();
  const [files, setFiles] = React.useState<Partial<InferFilesInput>>({});
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [result, setResult] = React.useState<InferResponse | null>(null);
  const [progress, setProgress] = React.useState<string>("");
  const [isExtractingZip, setIsExtractingZip] = React.useState(false);

  // Handle ZIP upload - use backend extraction
  const handleZipUpload = async (zipFile: File) => {
    setIsExtractingZip(true);
    setError(null);
    setResult(null);
    setProgress("");
    console.log("[ZIP] Starting ZIP upload and inference:", zipFile.name);
    
    setIsLoading(true);
    
    try {
      const response = await inferWithZip(
        zipFile,
        undefined, // patientId
        (stage, progressValue) => {
          const progressMsg = `${stage} (${Math.round(progressValue * 100)}%)`;
          console.log("[ZIP] Progress:", progressMsg);
          setProgress(progressMsg);
        }
      );

      console.log("[ZIP] ✅ Inference completed successfully!");
      console.log("[ZIP] Response received:", JSON.stringify(response, null, 2));
      setResult(response);
      setProgress("Complete");
      setError(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to process ZIP file";
      console.error("[ZIP] ❌ ZIP processing failed:", errorMessage, err);
      setError(errorMessage);
    } finally {
      setIsLoading(false);
      setIsExtractingZip(false);
    }
  };

  const handleFileChange = (name: keyof InferFilesInput, file: File | null) => {
    if (file) {
      // Check if it's a ZIP file
      if (file.name.toLowerCase().endsWith('.zip')) {
        handleZipUpload(file);
        return;
      }
      
      setFiles((prev) => ({ ...prev, [name]: file }));
    } else {
      setFiles((prev => {
        const newFiles = { ...prev };
        delete newFiles[name];
        return newFiles;
      }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setProgress("");

    console.log("[Form] Submit button clicked");
    console.log("[Form] Current files state:", {
      flair: files.flair?.name,
      t1: files.t1?.name,
      t1ce: files.t1ce?.name,
      t2: files.t2?.name,
      dist: files.dist?.name,
      boundary: files.boundary?.name,
    });

    // Validate required files
    if (!files.flair || !files.t1 || !files.t1ce || !files.t2) {
      const missing = [];
      if (!files.flair) missing.push("flair");
      if (!files.t1) missing.push("t1");
      if (!files.t1ce) missing.push("t1ce");
      if (!files.t2) missing.push("t2");
      const errorMsg = `Please upload all required files. Missing: ${missing.join(", ")}`;
      console.error("[Form] Validation failed:", errorMsg);
      setError(errorMsg);
      return;
    }

    console.log("[Form] Validation passed, starting inference...");
    setIsLoading(true);

    try {
      console.log("[Form] Calling inferWithFiles API...");
      const response = await inferWithFiles(
        {
          flair: files.flair!,
          t1: files.t1!,
          t1ce: files.t1ce!,
          t2: files.t2!,
          dist: files.dist,
          boundary: files.boundary,
        },
        (stage, progressValue) => {
          const progressMsg = `${stage} (${Math.round(progressValue * 100)}%)`;
          console.log("[Form] Progress:", progressMsg);
          setProgress(progressMsg);
        }
      );

      console.log("[Form] ✅ Inference completed successfully!");
      console.log("[Form] Response received:", JSON.stringify(response, null, 2));
      console.log("[Form] Output URLs:", response.outputs);
      console.log("[Form] Metrics:", response.metrics);
      setResult(response);
      setProgress("Complete");
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "An unknown error occurred";
      console.error("[Form] ❌ Inference failed:", errorMessage);
      console.error("[Form] Error details:", err);
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Upload Medical Imaging Files</CardTitle>
          <CardDescription>
            Upload FLAIR, T1, T1CE, and T2 MRI files for brain tumor segmentation
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* ZIP Upload Option */}
            <div className="space-y-2 p-4 border rounded-md bg-muted/50">
              <Label htmlFor="zip-upload" className="text-sm font-semibold">
                Quick Upload: ZIP File (Optional)
              </Label>
              <p className="text-xs text-muted-foreground mb-2">
                Upload a ZIP file containing flair, t1, t1ce, t2 files. They will be automatically extracted.
              </p>
              <Input
                id="zip-upload"
                type="file"
                accept=".zip"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    handleZipUpload(file);
                  }
                }}
                disabled={isLoading || isExtractingZip}
              />
              {isExtractingZip && (
                <p className="text-xs text-muted-foreground">Extracting ZIP file...</p>
              )}
            </div>

            {/* Required files */}
            <div className="space-y-2">
              <Label htmlFor="flair">FLAIR *</Label>
              <Input
                id="flair"
                type="file"
                accept=".nii,.nii.gz,.npy,.zip"
                onChange={(e) =>
                  handleFileChange("flair", e.target.files?.[0] || null)
                }
                disabled={isLoading || isExtractingZip}
                required
              />
              {files.flair && (
                <p className="text-xs text-muted-foreground">
                  Selected: {files.flair.name} ({(files.flair.size / 1024 / 1024).toFixed(2)} MB)
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="t1">T1 *</Label>
              <Input
                id="t1"
                type="file"
                accept=".nii,.nii.gz,.npy,.zip"
                onChange={(e) =>
                  handleFileChange("t1", e.target.files?.[0] || null)
                }
                disabled={isLoading || isExtractingZip}
                required
              />
              {files.t1 && (
                <p className="text-xs text-muted-foreground">
                  Selected: {files.t1.name} ({(files.t1.size / 1024 / 1024).toFixed(2)} MB)
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="t1ce">T1CE *</Label>
              <Input
                id="t1ce"
                type="file"
                accept=".nii,.nii.gz,.npy,.zip"
                onChange={(e) =>
                  handleFileChange("t1ce", e.target.files?.[0] || null)
                }
                disabled={isLoading || isExtractingZip}
                required
              />
              {files.t1ce && (
                <p className="text-xs text-muted-foreground">
                  Selected: {files.t1ce.name} ({(files.t1ce.size / 1024 / 1024).toFixed(2)} MB)
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="t2">T2 *</Label>
              <Input
                id="t2"
                type="file"
                accept=".nii,.nii.gz,.npy,.zip"
                onChange={(e) =>
                  handleFileChange("t2", e.target.files?.[0] || null)
                }
                disabled={isLoading || isExtractingZip}
                required
              />
              {files.t2 && (
                <p className="text-xs text-muted-foreground">
                  Selected: {files.t2.name} ({(files.t2.size / 1024 / 1024).toFixed(2)} MB)
                </p>
              )}
            </div>

            {/* Optional files */}
            <div className="space-y-2">
              <Label htmlFor="dist">Distance Map (optional)</Label>
              <Input
                id="dist"
                type="file"
                accept=".npy"
                onChange={(e) =>
                  handleFileChange("dist", e.target.files?.[0] || null)
                }
                disabled={isLoading}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="boundary">Boundary Map (optional)</Label>
              <Input
                id="boundary"
                type="file"
                accept=".npy"
                onChange={(e) =>
                  handleFileChange("boundary", e.target.files?.[0] || null)
                }
                disabled={isLoading}
              />
            </div>

            {error && (
              <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive border border-destructive">
                <strong>Error:</strong> {error}
                <p className="mt-2 text-xs">Check browser console (F12) for more details.</p>
              </div>
            )}

            {progress && (
              <div className="text-sm text-muted-foreground p-2 bg-muted rounded-md">
                <strong>Status:</strong> {progress}
              </div>
            )}

            {isLoading && (
              <div className="text-sm text-blue-600 p-2 bg-blue-50 rounded-md">
                Processing inference... This may take a few minutes.
              </div>
            )}

            <Button type="submit" disabled={isLoading} className="w-full">
              {isLoading ? "Processing..." : "Run Inference"}
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* Results */}
      {result && (
        <Card className="mt-6 border-2 border-green-500">
          <CardHeader>
            <CardTitle className="text-green-700">✅ Inference Results</CardTitle>
            <CardDescription>Segmentation outputs and metrics</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Metrics */}
            <div className="grid grid-cols-3 gap-4">
              <div className="rounded-lg border p-4">
                <div className="text-sm text-muted-foreground">Whole Tumor</div>
                <div className="text-2xl font-bold">
                  {result.metrics.wt_volume_cc.toFixed(2)} cc
                </div>
              </div>
              <div className="rounded-lg border p-4">
                <div className="text-sm text-muted-foreground">Tumor Core</div>
                <div className="text-2xl font-bold">
                  {result.metrics.tc_volume_cc.toFixed(2)} cc
                </div>
              </div>
              <div className="rounded-lg border p-4">
                <div className="text-sm text-muted-foreground">Enhancing Tumor</div>
                <div className="text-2xl font-bold">
                  {result.metrics.et_volume_cc.toFixed(2)} cc
                </div>
              </div>
            </div>

            {/* Montages */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Montages</h3>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                <div className="space-y-2">
                  <Label>Axial View</Label>
                  <div className="relative rounded-md border overflow-hidden bg-muted min-h-[200px]">
                    <ImageDisplay
                      src={result.outputs.montage_axial}
                      alt="Axial montage"
                      className="w-full h-auto"
                      onLoad={(e) => {
                        console.log(`[Image Load] Axial montage loaded successfully: ${result.outputs.montage_axial}`);
                        console.log(`[Image Load] Status: 200 OK`);
                      }}
                      onError={(e) => {
                        console.error(`[Image Error] Failed to load axial montage: ${result.outputs.montage_axial}`);
                      }}
                      loading="eager"
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Coronal View</Label>
                  <div className="relative rounded-md border overflow-hidden bg-muted min-h-[200px]">
                    <ImageDisplay
                      src={result.outputs.montage_coronal}
                      alt="Coronal montage"
                      className="w-full h-auto"
                      onLoad={(e) => {
                        console.log(`[Image Load] Coronal montage loaded successfully: ${result.outputs.montage_coronal}`);
                        console.log(`[Image Load] Status: 200 OK`);
                      }}
                      onError={(e) => {
                        console.error(`[Image Error] Failed to load coronal montage: ${result.outputs.montage_coronal}`);
                      }}
                      loading="eager"
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Sagittal View</Label>
                  <div className="relative rounded-md border overflow-hidden bg-muted min-h-[200px]">
                    <ImageDisplay
                      src={result.outputs.montage_sagittal}
                      alt="Sagittal montage"
                      className="w-full h-auto"
                      onLoad={(e) => {
                        console.log(`[Image Load] Sagittal montage loaded successfully: ${result.outputs.montage_sagittal}`);
                        console.log(`[Image Load] Status: 200 OK`);
                      }}
                      onError={(e) => {
                        console.error(`[Image Error] Failed to load sagittal montage: ${result.outputs.montage_sagittal}`);
                      }}
                      loading="eager"
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Class Overlays */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Class Overlays</h3>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                <div className="space-y-2">
                  <Label>Class 1 (NET)</Label>
                  <div className="relative rounded-md border overflow-hidden bg-muted min-h-[200px]">
                    <ImageDisplay
                      src={result.outputs.overlay_class_1}
                      alt="Class 1 overlay"
                      className="w-full h-auto"
                      onLoad={(e) => {
                        console.log(`[Image Load] Class 1 overlay loaded successfully: ${result.outputs.overlay_class_1}`);
                        console.log(`[Image Load] Status: 200 OK`);
                      }}
                      onError={(e) => {
                        console.error(`[Image Error] Failed to load class 1 overlay: ${result.outputs.overlay_class_1}`);
                      }}
                      loading="eager"
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Class 2 (ED)</Label>
                  <div className="relative rounded-md border overflow-hidden bg-muted min-h-[200px]">
                    <ImageDisplay
                      src={result.outputs.overlay_class_2}
                      alt="Class 2 overlay"
                      className="w-full h-auto"
                      onLoad={(e) => {
                        console.log(`[Image Load] Class 2 overlay loaded successfully: ${result.outputs.overlay_class_2}`);
                        console.log(`[Image Load] Status: 200 OK`);
                      }}
                      onError={(e) => {
                        console.error(`[Image Error] Failed to load class 2 overlay: ${result.outputs.overlay_class_2}`);
                      }}
                      loading="eager"
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Class 3 (ET)</Label>
                  <div className="relative rounded-md border overflow-hidden bg-muted min-h-[200px]">
                    <ImageDisplay
                      src={result.outputs.overlay_class_3}
                      alt="Class 3 overlay"
                      className="w-full h-auto"
                      onLoad={(e) => {
                        console.log(`[Image Load] Class 3 overlay loaded successfully: ${result.outputs.overlay_class_3}`);
                        console.log(`[Image Load] Status: 200 OK`);
                      }}
                      onError={(e) => {
                        console.error(`[Image Error] Failed to load class 3 overlay: ${result.outputs.overlay_class_3}`);
                      }}
                      loading="eager"
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Download Links */}
            <div className="space-y-2">
              <h3 className="text-lg font-semibold">Download Results</h3>
              <div className="flex flex-wrap gap-2">
                {result.outputs.segmentation_nifti && (
                  <Button
                    variant="outline"
                    asChild
                    size="sm"
                  >
                    <a
                      href={result.outputs.segmentation_nifti}
                      download
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Download NIfTI
                    </a>
                  </Button>
                )}
                <Button variant="outline" asChild size="sm">
                  <a
                    href={result.outputs.segmentation_numpy}
                    download
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Download NumPy
                  </a>
                </Button>
                <Button variant="outline" asChild size="sm">
                  <a
                    href={result.outputs.probability_maps}
                    download
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Download Probabilities
                  </a>
                </Button>
              </div>
            </div>

            {/* View 3D Tumour Button - Only show after successful inference */}
            {result.status === "success" && (
              <div className="pt-4 border-t">
                <Button
                  variant="default"
                  size="lg"
                  className="w-full"
                  onClick={() => {
                    // Store result in sessionStorage for the 3D page to access
                    sessionStorage.setItem(
                      `inference_result_${result.job_id}`,
                      JSON.stringify(result)
                    );
                    router.push(`/visualization/3d?job_id=${result.job_id}`);
                  }}
                >
                  View 3D Tumour
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* 3D Tumor Visualization */}
      {result && (result.outputs.mesh_necrotic || result.outputs.mesh_edema || result.outputs.mesh_enhancing) && (
        <Tumor3DViewer
          meshUrls={{
            necrotic: result.outputs.mesh_necrotic || null,
            edema: result.outputs.mesh_edema || null,
            enhancing: result.outputs.mesh_enhancing || null,
          }}
        />
      )}
    </div>
  );
}

