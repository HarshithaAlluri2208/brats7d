import type {
  InferFilesInput,
  InferResponse,
  UploadResponse,
  ProgressCallback,
} from "./types";

/**
 * Resolve API base URL at runtime in a way that avoids `process is not defined`
 * errors in the browser devtools console. Priority:
 * 1. window.__NEXT_PUBLIC_API_URL (if your app sets it)
 * 2. process.env.NEXT_PUBLIC_API_URL (Next.js build-time env)
 * 3. fallback to http://localhost:8000
 */
function getApiBase(): string {
  // 1) window-scoped override (optional, sometimes used in deployments)
  // @ts-ignore
  const winEnv = typeof window !== "undefined" ? (window as any).__NEXT_PUBLIC_API_URL : undefined;
  if (winEnv) {
    console.log("Using API base from window override:", winEnv);
    return winEnv;
  }

  // 2) Next.js / Node build-time env (available in client bundles when replaced at build time)
  // guard access to process to avoid ReferenceError in non-Node contexts
  const procEnv =
    typeof process !== "undefined" && process.env && process.env.NEXT_PUBLIC_API_URL
      ? process.env.NEXT_PUBLIC_API_URL
      : undefined;
  if (procEnv) {
    console.log("Using API base from process.env:", procEnv);
    return procEnv;
  }

  // 3) default fallback
  const fallback = "http://localhost:8000";
  console.log("API base not found in env, falling back to:", fallback);
  return fallback;
}

const API_BASE_URL = getApiBase();

export type InferResult = {
  mask_url: string; // URL to segmentation mask (could be presigned URL)
  wt_volume: number; // Whole Tumor (cc)
  tc_volume: number; // Tumor Core (cc)
  et_volume: number; // Enhancing Tumor (cc)
};

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Placeholder function for single file upload (used by dashboard)
// Note: The actual API requires multiple files (flair, t1, t1ce, t2)
// This is kept for backward compatibility with the dashboard component
export async function inferWithFile(file: File): Promise<InferResult> {
  // Mocked response for dashboard compatibility
  await sleep(1200); // simulate network + backend time
  return {
    mask_url: URL.createObjectURL(new Blob(["dummy-mask-nifti"], { type: "application/gzip" })),
    wt_volume: 48.2,
    tc_volume: 22.9,
    et_volume: 15.2,
  };
}

/**
 * Upload files and run inference on the backend.
 *
 * @param files - Object containing required files (flair, t1, t1ce, t2) and optional files (dist, boundary, patientId)
 * @param onProgress - Optional callback for progress updates (stage: string, progress: number)
 * @returns Promise resolving to the inference response with outputs and metrics
 * @throws Error if upload or inference fails
 */
export async function inferWithFiles(
  files: InferFilesInput,
  onProgress?: ProgressCallback
): Promise<InferResponse> {
  try {
    console.log("[API] inferWithFiles called");
    console.log("[API] Files to upload:", {
      flair: files.flair?.name,
      t1: files.t1?.name,
      t1ce: files.t1ce?.name,
      t2: files.t2?.name,
      dist: files.dist?.name,
      boundary: files.boundary?.name,
    });
    console.log("[API] API Base URL:", API_BASE_URL);

    // Step 1: Upload files
    onProgress?.("uploading", 0.2);

    const uploadFormData = new FormData();
    uploadFormData.append("flair", files.flair);
    uploadFormData.append("t1", files.t1);
    uploadFormData.append("t1ce", files.t1ce);
    uploadFormData.append("t2", files.t2);

    if (files.dist) {
      uploadFormData.append("dist", files.dist);
    }
    if (files.boundary) {
      uploadFormData.append("boundary", files.boundary);
    }
    if (files.patientId) {
      uploadFormData.append("patient_id", files.patientId);
    }

    const uploadUrl = `${API_BASE_URL}/infer/upload`;
    console.log("[API] Uploading to:", uploadUrl);
    console.log("[API] FormData entries:", Array.from(uploadFormData.entries()).map(([k, v]) => [k, v instanceof File ? v.name : v]));

    const uploadResponse = await fetch(uploadUrl, {
      method: "POST",
      body: uploadFormData,
    });

    console.log("[API] Upload response status:", uploadResponse.status, uploadResponse.statusText);

    if (!uploadResponse.ok) {
      const errorText = await uploadResponse.text();
      console.error("[API] Upload failed. Status:", uploadResponse.status);
      console.error("[API] Error response:", errorText);
      throw new Error(
        `Upload failed: ${uploadResponse.status} ${uploadResponse.statusText} - ${errorText}`
      );
    }

    const uploadData: UploadResponse = await uploadResponse.json();
    console.log("[API] Upload successful. Response:", uploadData);
    const jobId = uploadData.job_id;

    if (!jobId) {
      console.error("[API] Upload response missing job_id:", uploadData);
      throw new Error("Upload response missing job_id");
    }

    console.log("[API] Job ID received:", jobId);
    onProgress?.("processing", 0.5);

    // Step 2: Run inference
    const runFormData = new FormData();
    runFormData.append("job_id", jobId);

    const runUrl = `${API_BASE_URL}/infer/run`;
    console.log("[API] Triggering inference at:", runUrl, "with job_id:", jobId);
    const runResponse = await fetch(runUrl, {
      method: "POST",
      body: runFormData,
    });

    console.log("[API] Inference response status:", runResponse.status, runResponse.statusText);

    if (!runResponse.ok) {
      const errorText = await runResponse.text();
      console.error("[API] Inference failed. Status:", runResponse.status);
      console.error("[API] Error response:", errorText);
      throw new Error(
        `Inference failed: ${runResponse.status} ${runResponse.statusText} - ${errorText}`
      );
    }

    onProgress?.("complete", 1.0);

    const inferData: InferResponse = await runResponse.json();
    console.log("[API] ✅ Inference completed successfully!");
    console.log("[API] Full response:", JSON.stringify(inferData, null, 2));
    return inferData;
  } catch (error) {
    console.error("[API] ❌ Error in inferWithFiles:", error);
    if (error instanceof Error) {
      throw error;
    }
    throw new Error(`Unknown error during inference: ${String(error)}`);
  }
}

/**
 * Upload a ZIP file and run inference on the backend.
 * The backend will extract and validate the ZIP file.
 *
 * @param zipFile - ZIP file containing NIfTI files (flair, t1, t1ce, t2, seg, boundary, distance)
 * @param patientId - Optional patient ID
 * @param onProgress - Optional callback for progress updates (stage: string, progress: number)
 * @returns Promise resolving to the inference response with outputs and metrics
 * @throws Error if upload or inference fails
 */
export async function inferWithZip(
  zipFile: File,
  patientId?: string,
  onProgress?: ProgressCallback
): Promise<InferResponse> {
  try {
    console.log("[API] inferWithZip called");
    console.log("[API] ZIP file:", zipFile.name, `(${(zipFile.size / 1024 / 1024).toFixed(2)} MB)`);
    console.log("[API] API Base URL:", API_BASE_URL);

    // Step 1: Upload ZIP file
    onProgress?.("uploading", 0.2);

    const uploadFormData = new FormData();
    uploadFormData.append("zip_file", zipFile);
    if (patientId) {
      uploadFormData.append("patient_id", patientId);
    }

    const uploadUrl = `${API_BASE_URL}/infer/upload-zip`;
    console.log("[API] Uploading ZIP to:", uploadUrl);

    const uploadResponse = await fetch(uploadUrl, {
      method: "POST",
      body: uploadFormData,
    });

    console.log("[API] Upload response status:", uploadResponse.status, uploadResponse.statusText);

    if (!uploadResponse.ok) {
      const errorText = await uploadResponse.text();
      console.error("[API] ZIP upload failed. Status:", uploadResponse.status);
      console.error("[API] Error response:", errorText);
      throw new Error(
        `ZIP upload failed: ${uploadResponse.status} ${uploadResponse.statusText} - ${errorText}`
      );
    }

    const uploadData: UploadResponse = await uploadResponse.json();
    console.log("[API] ZIP upload successful. Response:", uploadData);
    const jobId = uploadData.job_id;

    if (!jobId) {
      console.error("[API] Upload response missing job_id:", uploadData);
      throw new Error("Upload response missing job_id");
    }

    console.log("[API] Job ID received:", jobId);
    onProgress?.("processing", 0.5);

    // Step 2: Run inference
    const runFormData = new FormData();
    runFormData.append("job_id", jobId);

    const runUrl = `${API_BASE_URL}/infer/run`;
    console.log("[API] Triggering inference at:", runUrl, "with job_id:", jobId);
    const runResponse = await fetch(runUrl, {
      method: "POST",
      body: runFormData,
    });

    console.log("[API] Inference response status:", runResponse.status, runResponse.statusText);

    if (!runResponse.ok) {
      const errorText = await runResponse.text();
      console.error("[API] Inference failed. Status:", runResponse.status);
      console.error("[API] Error response:", errorText);
      throw new Error(
        `Inference failed: ${runResponse.status} ${runResponse.statusText} - ${errorText}`
      );
    }

    onProgress?.("complete", 1.0);

    const inferData: InferResponse = await runResponse.json();
    console.log("[API] ✅ Inference completed successfully!");
    console.log("[API] Full response:", JSON.stringify(inferData, null, 2));
    return inferData;
  } catch (error) {
    console.error("[API] ❌ Error in inferWithZip:", error);
    if (error instanceof Error) {
      throw error;
    }
    throw new Error(`Unknown error during ZIP inference: ${String(error)}`);
  }
}