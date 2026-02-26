"use client";

import * as React from "react";
import { inferWithFiles } from "@/lib/api";
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

interface FileUploadFormProps {
  onComplete?: (results: InferResponse) => void;
}

const ALLOWED_EXTENSIONS = [".nii", ".nii.gz", ".npy"];

function validateFileType(file: File): boolean {
  const fileName = file.name.toLowerCase();
  return ALLOWED_EXTENSIONS.some((ext) => fileName.endsWith(ext));
}

function getFileExtension(file: File): string {
  const fileName = file.name.toLowerCase();
  if (fileName.endsWith(".nii.gz")) return ".nii.gz";
  if (fileName.endsWith(".nii")) return ".nii";
  if (fileName.endsWith(".npy")) return ".npy";
  return "";
}

export function FileUploadForm({ onComplete }: FileUploadFormProps) {
  const [files, setFiles] = React.useState<Partial<InferFilesInput>>({});
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [progress, setProgress] = React.useState<string>("");
  const [progressValue, setProgressValue] = React.useState(0);

  const handleFileChange = (
    name: keyof InferFilesInput,
    file: File | null,
    event?: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (file) {
      // Validate file type
      if (!validateFileType(file)) {
        setError(
          `Invalid file type for ${name}. Allowed types: ${ALLOWED_EXTENSIONS.join(", ")}`
        );
        if (event) {
          event.target.value = ""; // Clear invalid file
        }
        return;
      }

      // Additional validation for optional files
      if ((name === "dist" || name === "boundary") && !file.name.endsWith(".npy")) {
        setError(`${name} must be a .npy file`);
        if (event) {
          event.target.value = "";
        }
        return;
      }

      setFiles((prev) => ({ ...prev, [name]: file }));
      setError(null); // Clear error on valid file
    } else {
      setFiles((prev) => {
        const newFiles = { ...prev };
        delete newFiles[name];
        return newFiles;
      });
    }
  };

  const handlePatientIdChange = (value: string) => {
    setFiles((prev) => ({ ...prev, patientId: value || undefined }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setProgress("");
    setProgressValue(0);

    // Validate required files
    if (!files.flair || !files.t1 || !files.t1ce || !files.t2) {
      setError("Please upload all required files: FLAIR, T1, T1CE, and T2");
      return;
    }

    // Validate file types again
    const requiredFiles = [files.flair, files.t1, files.t1ce, files.t2];
    for (const file of requiredFiles) {
      if (file && !validateFileType(file)) {
        setError(`Invalid file type: ${file.name}. Allowed: ${ALLOWED_EXTENSIONS.join(", ")}`);
        return;
      }
    }

    setIsLoading(true);

    try {
      const response = await inferWithFiles(
        {
          flair: files.flair!,
          t1: files.t1!,
          t1ce: files.t1ce!,
          t2: files.t2!,
          dist: files.dist,
          boundary: files.boundary,
          patientId: files.patientId,
        },
        (stage, progressNum) => {
          setProgress(stage);
          setProgressValue(progressNum);
        }
      );

      // Call parent callback with results
      if (onComplete) {
        onComplete(response);
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : "An unknown error occurred during inference";
      setError(errorMessage);
    } finally {
      setIsLoading(false);
      setProgress("");
      setProgressValue(0);
    }
  };

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle>Upload Medical Imaging Files</CardTitle>
        <CardDescription>
          Upload FLAIR, T1, T1CE, and T2 MRI files for brain tumor segmentation.
          Distance and boundary maps are optional.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Patient ID (optional) */}
          <div className="space-y-2">
            <Label htmlFor="patientId">Patient ID (optional)</Label>
            <Input
              id="patientId"
              type="text"
              placeholder="Enter patient identifier"
              value={files.patientId || ""}
              onChange={(e) => handlePatientIdChange(e.target.value)}
              disabled={isLoading}
              aria-describedby="patientId-description"
            />
            <p
              id="patientId-description"
              className="text-xs text-muted-foreground"
            >
              Optional identifier for tracking this patient
            </p>
          </div>

          {/* Required files */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold">Required Files *</h3>

            <div className="space-y-2">
              <Label htmlFor="flair">
                FLAIR * <span className="text-muted-foreground text-xs">(.nii, .nii.gz, .npy)</span>
              </Label>
              <Input
                id="flair"
                type="file"
                accept=".nii,.nii.gz,.npy"
                onChange={(e) =>
                  handleFileChange("flair", e.target.files?.[0] || null, e)
                }
                disabled={isLoading}
                required
                aria-describedby="flair-description"
                aria-invalid={error?.includes("flair") ? "true" : "false"}
              />
              {files.flair && (
                <p className="text-xs text-muted-foreground">
                  Selected: {files.flair.name} ({(files.flair.size / 1024 / 1024).toFixed(2)} MB)
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="t1">
                T1 * <span className="text-muted-foreground text-xs">(.nii, .nii.gz, .npy)</span>
              </Label>
              <Input
                id="t1"
                type="file"
                accept=".nii,.nii.gz,.npy"
                onChange={(e) =>
                  handleFileChange("t1", e.target.files?.[0] || null, e)
                }
                disabled={isLoading}
                required
                aria-describedby="t1-description"
              />
              {files.t1 && (
                <p className="text-xs text-muted-foreground">
                  Selected: {files.t1.name} ({(files.t1.size / 1024 / 1024).toFixed(2)} MB)
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="t1ce">
                T1CE * <span className="text-muted-foreground text-xs">(.nii, .nii.gz, .npy)</span>
              </Label>
              <Input
                id="t1ce"
                type="file"
                accept=".nii,.nii.gz,.npy"
                onChange={(e) =>
                  handleFileChange("t1ce", e.target.files?.[0] || null, e)
                }
                disabled={isLoading}
                required
                aria-describedby="t1ce-description"
              />
              {files.t1ce && (
                <p className="text-xs text-muted-foreground">
                  Selected: {files.t1ce.name} ({(files.t1ce.size / 1024 / 1024).toFixed(2)} MB)
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="t2">
                T2 * <span className="text-muted-foreground text-xs">(.nii, .nii.gz, .npy)</span>
              </Label>
              <Input
                id="t2"
                type="file"
                accept=".nii,.nii.gz,.npy"
                onChange={(e) =>
                  handleFileChange("t2", e.target.files?.[0] || null, e)
                }
                disabled={isLoading}
                required
                aria-describedby="t2-description"
              />
              {files.t2 && (
                <p className="text-xs text-muted-foreground">
                  Selected: {files.t2.name} ({(files.t2.size / 1024 / 1024).toFixed(2)} MB)
                </p>
              )}
            </div>
          </div>

          {/* Optional files */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold">Optional Files</h3>

            <div className="space-y-2">
              <Label htmlFor="dist">
                Distance Map <span className="text-muted-foreground text-xs">(.npy)</span>
              </Label>
              <Input
                id="dist"
                type="file"
                accept=".npy"
                onChange={(e) =>
                  handleFileChange("dist", e.target.files?.[0] || null, e)
                }
                disabled={isLoading}
                aria-describedby="dist-description"
              />
              {files.dist && (
                <p className="text-xs text-muted-foreground">
                  Selected: {files.dist.name} ({(files.dist.size / 1024 / 1024).toFixed(2)} MB)
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="boundary">
                Boundary Map <span className="text-muted-foreground text-xs">(.npy)</span>
              </Label>
              <Input
                id="boundary"
                type="file"
                accept=".npy"
                onChange={(e) =>
                  handleFileChange("boundary", e.target.files?.[0] || null, e)
                }
                disabled={isLoading}
                aria-describedby="boundary-description"
              />
              {files.boundary && (
                <p className="text-xs text-muted-foreground">
                  Selected: {files.boundary.name} ({(files.boundary.size / 1024 / 1024).toFixed(2)} MB)
                </p>
              )}
            </div>
          </div>

          {/* Error message */}
          {error && (
            <div
              role="alert"
              className="rounded-md bg-destructive/10 p-3 text-sm text-destructive border border-destructive/20"
              aria-live="polite"
            >
              <strong>Error:</strong> {error}
            </div>
          )}

          {/* Progress indicator */}
          {isLoading && (
            <div className="space-y-2" role="status" aria-live="polite">
              <div className="flex items-center gap-2">
                <div
                  className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent"
                  aria-hidden="true"
                />
                <span className="text-sm text-muted-foreground">
                  {progress || "Processing..."}
                </span>
              </div>
              {progressValue > 0 && (
                <div className="w-full bg-muted rounded-full h-2">
                  <div
                    className="bg-primary h-2 rounded-full transition-all duration-300"
                    style={{ width: `${progressValue * 100}%` }}
                    role="progressbar"
                    aria-valuenow={Math.round(progressValue * 100)}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  />
                </div>
              )}
            </div>
          )}

          {/* Submit button */}
          <Button
            type="submit"
            disabled={isLoading}
            className="w-full"
            aria-busy={isLoading}
          >
            {isLoading ? (
              <>
                <div
                  className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent"
                  aria-hidden="true"
                />
                Processing...
              </>
            ) : (
              "Run Inference"
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

