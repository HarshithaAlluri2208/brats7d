export function SiteFooter() {
  return (
    <footer className="border-t py-6 text-sm text-muted-foreground">
      <div className="container mx-auto px-6 flex items-center justify-between">
        <p>© {new Date().getFullYear()} NeuroVision</p>
        <p className="hidden sm:block">Brain tumor segmentation demo UI</p>
      </div>
    </footer>
  );
}


