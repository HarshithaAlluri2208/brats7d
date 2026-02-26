"use client";

import * as React from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Tumor3DFromNifti } from "@/components/tumor-3d-from-nifti";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import type { InferResponse } from "@/lib/types";

export default function Visualization3DPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const jobId = searchParams.get("job_id");
  const [result, setResult] = React.useState<InferResponse | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!jobId) {
      setError("No job ID provided");
      setLoading(false);
      return;
    }

    // Try to get result from sessionStorage first
    const storedResult = sessionStorage.getItem(`inference_result_${jobId}`);
    if (storedResult) {
      try {
        const parsed = JSON.parse(storedResult) as InferResponse;
        setResult(parsed);
        setLoading(false);
        return;
      } catch (e) {
        console.error("Failed to parse stored result:", e);
      }
    }

    // If not in sessionStorage, show error
    setError("Inference results not found. Please run inference first.");
    setLoading(false);
  }, [jobId]);

  const hasSegNifti = result && result.outputs.segmentation_nifti;

  if (loading) {
    return (
      <div className="container mx-auto px-6 py-10">
        <Card>
          <CardContent className="py-10 text-center">
            <p className="text-muted-foreground">Loading 3D visualization...</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="container mx-auto px-6 py-10">
        <Card>
          <CardHeader>
            <CardTitle>Error</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-destructive">{error || "Inference results not found"}</p>
            <Button onClick={() => router.push("/dashboard")} variant="outline">
              Go to Dashboard
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!hasSegNifti) {
    return (
      <div className="container mx-auto px-6 py-10">
        <Card>
          <CardHeader>
            <CardTitle>3D Visualization Unavailable</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-muted-foreground">
              No segmentation NIfTI file available for this inference. The segmentation
              file (seg.nii) is required for 3D visualization.
            </p>
            <Button onClick={() => router.push("/dashboard")} variant="outline">
              Go to Dashboard
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-6 py-10">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl md:text-3xl font-semibold">3D Tumor Visualization</h1>
          <p className="mt-2 text-muted-foreground">
            Interactive 3D view of tumor subregions from inference job: {jobId}
          </p>
        </div>
        <Button onClick={() => router.push("/dashboard")} variant="outline">
          Back to Dashboard
        </Button>
      </div>

      <Tumor3DFromNifti segNiftiUrl={result.outputs.segmentation_nifti!} />

      {/* Metrics Summary */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Inference Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
        </CardContent>
      </Card>
    </div>
  );
}
