/**
 * Type definitions for inference API responses
 */

export type InferFilesInput = {
  flair: File;
  t1: File;
  t1ce: File;
  t2: File;
  dist?: File;
  boundary?: File;
  patientId?: string;
};

export type UploadResponse = {
  job_id: string;
  status: string;
  files: Record<string, string>;
};

export type InferOutputs = {
  montage_axial: string;
  montage_coronal: string;
  montage_sagittal: string;
  overlay_class_1: string;
  overlay_class_2: string;
  overlay_class_3: string;
  comparison_overlay?: string | null;
  segmentation_nifti: string | null;
  segmentation_numpy: string;
  probability_maps: string;
  mesh_necrotic?: string | null;
  mesh_edema?: string | null;
  mesh_enhancing?: string | null;
};

export type InferMetrics = {
  wt_volume_cc: number;
  tc_volume_cc: number;
  et_volume_cc: number;
};

export type InferResponse = {
  job_id: string;
  status: string;
  outputs: InferOutputs;
  metrics: InferMetrics;
};

export type ProgressCallback = (stage: string, progress: number) => void;

