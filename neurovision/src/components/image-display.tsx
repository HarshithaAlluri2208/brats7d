"use client";

import * as React from "react";

interface ImageDisplayProps {
  src: string;
  alt: string;
  className?: string;
  style?: React.CSSProperties;
  onLoad?: (e: React.SyntheticEvent<HTMLImageElement, Event>) => void;
  onError?: (e: React.SyntheticEvent<HTMLImageElement, Event>) => void;
  loading?: "lazy" | "eager";
}

/**
 * Robust image display component that handles:
 * - Direct image URLs
 * - Blob URLs for images served with Content-Disposition: attachment
 * - Proper cleanup of object URLs
 */
export function ImageDisplay({
  src,
  alt,
  className,
  style,
  onLoad,
  onError,
  loading = "lazy",
}: ImageDisplayProps) {
  const [imageSrc, setImageSrc] = React.useState<string | null>(null);
  const [isLoading, setIsLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const blobUrlRef = React.useRef<string | null>(null);

  // Cleanup blob URL on unmount
  React.useEffect(() => {
    return () => {
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, []);

  // Always fetch as blob to prevent downloads
  React.useEffect(() => {
    if (!src) {
      setIsLoading(false);
      setError("No image source provided");
      return;
    }

    setIsLoading(true);
    setError(null);
    
    // Cleanup previous blob URL if it exists
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }

    // Always fetch as blob to prevent browser download behavior
    const fetchImage = async () => {
      try {
        console.log(`[ImageDisplay] Fetching image as blob: ${src}`);
        const response = await fetch(src, {
          method: "GET",
          credentials: "include",
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const blob = await response.blob();
        
        // Verify blob is actually an image
        if (!blob.type.startsWith("image/")) {
          throw new Error(`Blob is not an image type: ${blob.type}`);
        }

        const blobUrl = URL.createObjectURL(blob);
        blobUrlRef.current = blobUrl;
        setImageSrc(blobUrl);
        setIsLoading(false);
        setError(null);
        
        console.log(`[ImageDisplay] Successfully created blob URL for: ${src}`);
        console.log(`[ImageDisplay] Blob type: ${blob.type}, size: ${blob.size} bytes`);
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Failed to load image";
        console.error(`[ImageDisplay] Blob fetch failed: ${errorMessage}`, err);
        setError(errorMessage);
        setIsLoading(false);
        setImageSrc(null);
      }
    };

    fetchImage();
  }, [src]);

  const handleImageError = React.useCallback(
    (e: React.SyntheticEvent<HTMLImageElement, Event>) => {
      // Image should already be loaded as blob, so if this fires, it's a real error
      console.error(`[ImageDisplay] Image load error for blob URL:`, e);
      setError("Failed to display image");
      setIsLoading(false);
      if (onError) {
        onError(e);
      }
    },
    [onError]
  );

  const handleImageLoad = React.useCallback(
    (e: React.SyntheticEvent<HTMLImageElement, Event>) => {
      setIsLoading(false);
      setError(null);
      console.log(`[ImageDisplay] Image loaded successfully: ${src}`);
      if (onLoad) {
        onLoad(e);
      }
    },
    [src, onLoad]
  );

  return (
    <div className="relative w-full h-full">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-muted z-10">
          <div className="text-sm text-muted-foreground">Loading...</div>
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-muted z-10">
          <div className="text-sm text-destructive">Failed to load image</div>
        </div>
      )}
      {imageSrc && (
        <img
          src={imageSrc}
          alt={alt}
          className={className}
          onLoad={handleImageLoad}
          onError={handleImageError}
          loading={loading}
          style={{ display: error ? "none" : "block", width: "100%", height: "auto", ...style }}
        />
      )}
    </div>
  );
}

