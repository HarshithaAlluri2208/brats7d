"use client";

import { useCallback, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import Link from "next/link";
import { inferWithFile } from "@/lib/api";
import { FileUploadForm } from "@/components/FileUploadForm";

type InferResponse = {
  shape: [number, number, number];
  maskNonZeroVoxels: number;
};

export default function DashboardPage() {
  const [fileName, setFileName] = useState<string>("");
  const [fileObj, setFileObj] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [phase, setPhase] = useState<"idle" | "upload" | "process" | "done">("idle");
  const [result, setResult] = useState<InferResponse | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback((file: File) => {
    const valid = /\.(nii\.gz|zip)$/i.test(file.name);
    if (!valid) {
      alert("Please upload a .nii.gz or .zip file.");
      return;
    }
    setFileName(file.name);
    setFileObj(file);
  }, []);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  }

  function onDragOver(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }
  function onDragLeave(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }
  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFileSelect(file);
  }

  async function simulateUpload(): Promise<void> {
    setPhase("upload");
    setProgress(5);
    for (const step of [15, 30, 45, 60, 75, 90, 100]) {
      await new Promise((r) => setTimeout(r, 250));
      setProgress(step);
    }
  }

  async function handleInfer(): Promise<InferResponse> {
    setPhase("process");
    setProgress(10);
    for (const step of [25, 42, 58, 73, 88, 100]) {
      await new Promise((r) => setTimeout(r, 250));
      setProgress(step);
    }
    // Use placeholder API client; map to dashboard result for continuity
    const apiRes = await inferWithFile(fileObj as File);
    return { shape: [256, 256, 256], maskNonZeroVoxels: Math.round(apiRes.wt_volume * 1000) };
  }

  async function startPipeline() {
    if (!fileName) return;
    setResult(null);
    await simulateUpload();
    const res = await handleInfer();
    setResult(res);
    setPhase("done");
  }

  const stepClass = (active: boolean) => active ? "text-foreground" : "text-muted-foreground";

  return (
    <div className="container mx-auto px-6 py-10">
      <motion.h2 initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="text-2xl md:text-3xl font-semibold">
        MRI Upload & Inference
      </motion.h2>
      <p className="mt-2 text-muted-foreground">Upload a brain MRI archive (.nii.gz or .zip). We’ll process it and let you visualize results.</p>

      <div className="mt-6">
        {/* Use the full FileUploadForm component with results display */}
        <FileUploadForm />
      </div>
    </div>
  );
}


