import React, { useEffect, useRef } from 'react';
import { cn } from '../lib/utils';

interface OverlayProps {
  imageUrl: string;
  analysis: any;
  showDish?: boolean;
  showMask?: boolean;
  showCracks?: boolean;
  showROI?: boolean;
  className?: string;
}

export const AnalysisOverlay: React.FC<OverlayProps> = ({
  imageUrl,
  analysis,
  showDish = true,
  showMask = true,
  showCracks = true,
  showROI = true,
  className
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !analysis) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      const scaleX = img.width / 1000;
      const scaleY = img.height / 1000;

      // 1. Dish Detection Overlay
      if (showDish && analysis.dish_center) {
        ctx.beginPath();
        ctx.arc(
          analysis.dish_center.x * scaleX,
          analysis.dish_center.y * scaleY,
          analysis.dish_radius * scaleX,
          0,
          Math.PI * 2
        );
        ctx.strokeStyle = '#3b82f6'; // blue-500
        ctx.lineWidth = 4;
        ctx.stroke();
        ctx.fillStyle = 'rgba(59, 130, 246, 0.1)';
        ctx.fill();
      }

      // 2. Segmentation Mask
      if (showMask && analysis.colony_polygon) {
        ctx.beginPath();
        analysis.colony_polygon.forEach((p: any, i: number) => {
          if (i === 0) ctx.moveTo(p.x * scaleX, p.y * scaleY);
          else ctx.lineTo(p.x * scaleX, p.y * scaleY);
        });
        ctx.closePath();
        ctx.strokeStyle = '#10b981'; // emerald-500
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.fillStyle = 'rgba(16, 185, 129, 0.3)';
        ctx.fill();
      }

      // 3. Cracks Overlay
      if (showCracks && analysis.cracks) {
        analysis.cracks.forEach((crack: any[]) => {
          ctx.beginPath();
          crack.forEach((p, i) => {
            if (i === 0) ctx.moveTo(p.x * scaleX, p.y * scaleY);
            else ctx.lineTo(p.x * scaleX, p.y * scaleY);
          });
          ctx.strokeStyle = '#f59e0b'; // amber-500
          ctx.lineWidth = 2;
          ctx.stroke();
        });
      }

      // 4. ROI (Bounding Box of colony)
      if (showROI && analysis.colony_polygon) {
        const xs = analysis.colony_polygon.map((p: any) => p.x);
        const ys = analysis.colony_polygon.map((p: any) => p.y);
        const minX = Math.min(...xs) * scaleX;
        const maxX = Math.max(...xs) * scaleX;
        const minY = Math.min(...ys) * scaleY;
        const maxY = Math.max(...ys) * scaleY;

        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = '#ef4444'; // red-500
        ctx.lineWidth = 2;
        ctx.strokeRect(minX - 10, minY - 10, (maxX - minX) + 20, (maxY - minY) + 20);
        ctx.setLineDash([]);
      }
    };
    img.src = imageUrl;
  }, [imageUrl, analysis, showDish, showMask, showCracks, showROI]);

  return (
    <canvas 
      ref={canvasRef} 
      className={cn("max-w-full max-h-full object-contain", className)} 
    />
  );
};
