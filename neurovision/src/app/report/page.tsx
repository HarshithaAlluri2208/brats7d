"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

const metrics = [
  { key: "WT", label: "Whole Tumor", value: "48.2 cm³", color: "bg-rose-500" },
  { key: "TC", label: "Tumor Core", value: "22.9 cm³", color: "bg-amber-400" },
  { key: "ET", label: "Enhancing Tumor", value: "15.2 cm³", color: "bg-emerald-400" },
  { key: "DSC", label: "Dice (WT)", value: "0.87", color: "bg-sky-400" },
];

function downloadBlob(filename: string, mime: string, content: string) {
  const blob = new Blob([content], { type: mime });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

export default function ReportPage() {
  function handleDownloadMask() {
    // Placeholder: in real app, fetch binary from backend
    downloadBlob("neurovision-mask.nii.gz", "application/gzip", "Dummy compressed NIfTI mask placeholder");
  }

  function handleDownloadPdf() {
    // Placeholder: a very simple text-as-PDF mime; replace with real PDF generation
    downloadBlob(
      "neurovision-report.pdf",
      "application/pdf",
      "NeuroVision Report\n\nEnhancing Tumor Volume: 15.2 cc\nTumor Core: 22.9 cc\nWhole Tumor: 48.2 cc"
    );
  }

  return (
    <div className="pb-12">
      {/* Gradient header */}
      <section className="relative overflow-hidden">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="container mx-auto px-6 py-12"
        >
          <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">Results & Report</h1>
          <p className="mt-2 text-muted-foreground max-w-2xl">
            Summary of segmentation findings with export options for clinical review.
          </p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.18 }}
          transition={{ duration: 1.2 }}
          className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(70rem_70rem_at_50%_-10%,hsl(0_0%_100%/10%),transparent_60%)]"
        />
      </section>

      {/* Content */}
      <div className="container mx-auto px-6">
        <div className="grid gap-6 md:grid-cols-3">
          {metrics.map((m) => (
            <motion.div key={m.key} initial={{ opacity: 0, y: 8 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <span className={`inline-block h-3 w-3 rounded-sm ${m.color}`} /> {m.label}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-lg font-medium">{m.value}</p>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        <motion.div initial={{ opacity: 0, y: 8 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Summary</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-muted-foreground">
              <p>
                Enhancing Tumor Volume: <span className="text-foreground font-medium">15.2 cc</span>
              </p>
              <p>
                Tumor Core: <span className="text-foreground font-medium">22.9 cc</span>
              </p>
              <p>
                Whole Tumor: <span className="text-foreground font-medium">48.2 cc</span>
              </p>
              <p>
                Findings are consistent with a high-grade glioma pattern. Correlate clinically.
              </p>
              <div className="pt-2 flex flex-wrap gap-3">
                <Button onClick={handleDownloadMask}>Download mask (.nii.gz)</Button>
                <Button variant="secondary" onClick={handleDownloadPdf}>Download PDF report</Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}


