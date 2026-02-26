"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

function generateMockSlice(seed = 128, size = 256) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const r = size/2;
  const shift = (seed % 64) - 32; // subtle variation across slices
  const grd = ctx.createRadialGradient(size/2 + shift, size/2 - shift, 20, size/2, size/2, r);
  grd.addColorStop(0, "#8b5cf6");
  grd.addColorStop(1, "#0ea5e9");
  ctx.fillStyle = grd;
  ctx.fillRect(0,0,size,size);
  return canvas.toDataURL();
}

function generateMockMask(seed = 128, size = 256) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "rgba(244,63,94,0.85)"; // rose-500 w/ alpha
  ctx.beginPath();
  const radius = 40 + (seed % 30);
  ctx.arc(size/2 + 20, size/2 - 12, radius, 0, Math.PI * 2);
  ctx.fill();
  return canvas.toDataURL();
}

export default function VisualizationPage() {
  const [index, setIndex] = useState<number>(128);
  const [overlay, setOverlay] = useState<boolean>(true);
  const [base, setBase] = useState<string>("");
  const [mask, setMask] = useState<string>("");
  const [overlayOpacity, setOverlayOpacity] = useState<number>(70);

  // Generate images only after mount to avoid SSR/client mismatch
  useEffect(() => {
    setBase(generateMockSlice(index));
    setMask(generateMockMask(index));
  }, [index]);

  function onWheel(e: React.WheelEvent<HTMLDivElement>) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 1 : -1;
    setIndex((prev) => Math.min(255, Math.max(0, prev + delta)));
  }

  return (
    <div className="container mx-auto px-6 py-10">
      <motion.h2 initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="text-2xl md:text-3xl font-semibold">
        Visualization
      </motion.h2>
      <p className="mt-2 text-muted-foreground">Scroll through slices and toggle tumor overlay.</p>

      <div className="mt-6 grid gap-6 md:grid-cols-[1fr_320px]">
        <Card>
          <CardHeader>
            <CardTitle>Slice Viewer</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col items-center gap-4">
            <div onWheel={onWheel} className="relative rounded-lg overflow-hidden border" style={{ width: 360, height: 360 }}>
              {base && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={base} alt="MRI slice" className="absolute inset-0 h-full w-full object-cover" />
              )}
              {overlay && mask && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={mask} alt="Segmentation mask" className="absolute inset-0 h-full w-full object-cover mix-blend-screen" style={{ opacity: overlayOpacity / 100 }} />
              )}
            </div>
            <div className="w-full space-y-2">
              <Label htmlFor="slice">Slice index: {index}</Label>
              <Slider id="slice" value={[index]} min={0} max={255} step={1} onValueChange={(v) => setIndex(v[0])} />
            </div>
            <div className="w-full space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="opacity">Overlay opacity</Label>
                <span className="text-xs text-muted-foreground">{overlayOpacity}%</span>
              </div>
              <Slider id="opacity" value={[overlayOpacity]} min={0} max={100} step={1} onValueChange={(v) => setOverlayOpacity(v[0])} />
            </div>
          </CardContent>
        </Card>

        <div className="grid gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Overlay</CardTitle>
            </CardHeader>
            <CardContent className="flex items-center gap-3">
              <Switch id="overlay" checked={overlay} onCheckedChange={setOverlay} />
              <Label htmlFor="overlay">Show segmentation mask</Label>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Tumor Stats</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2"><span className="inline-block h-3 w-3 rounded-sm bg-rose-500" /> WT</div>
                <span className="text-muted-foreground">48.2 cm³</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2"><span className="inline-block h-3 w-3 rounded-sm bg-amber-400" /> TC</div>
                <span className="text-muted-foreground">22.9 cm³</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2"><span className="inline-block h-3 w-3 rounded-sm bg-emerald-400" /> ET</div>
                <span className="text-muted-foreground">9.7 cm³</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}


