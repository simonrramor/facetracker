// Color Matcher
// Adjusts color and lighting of source face texture to match live video

export class ColorMatcher {
  constructor() {
    this.sourceStats = null;
    this.targetStats = null;
  }

  // Analyze color statistics of an image/canvas
  analyzeColors(canvas) {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    let rSum = 0, gSum = 0, bSum = 0;
    let rSqSum = 0, gSqSum = 0, bSqSum = 0;
    let count = 0;

    // Only analyze non-transparent pixels
    for (let i = 0; i < data.length; i += 4) {
      const alpha = data[i + 3];
      if (alpha > 128) { // Only count visible pixels
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];

        rSum += r;
        gSum += g;
        bSum += b;

        rSqSum += r * r;
        gSqSum += g * g;
        bSqSum += b * b;

        count++;
      }
    }

    if (count === 0) {
      return { meanR: 128, meanG: 128, meanB: 128, stdR: 50, stdG: 50, stdB: 50 };
    }

    const meanR = rSum / count;
    const meanG = gSum / count;
    const meanB = bSum / count;

    const stdR = Math.sqrt(rSqSum / count - meanR * meanR);
    const stdG = Math.sqrt(gSqSum / count - meanG * meanG);
    const stdB = Math.sqrt(bSqSum / count - meanB * meanB);

    return { meanR, meanG, meanB, stdR, stdG, stdB };
  }

  // Analyze colors from a video frame using face landmarks
  analyzeVideoFrame(video, landmarks, width, height) {
    // Create a temporary canvas to extract face region
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const ctx = tempCanvas.getContext('2d');

    // Draw video frame
    ctx.drawImage(video, 0, 0, width, height);

    // Create a mask for the face region
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = width;
    maskCanvas.height = height;
    const maskCtx = maskCanvas.getContext('2d');

    // Draw face outline
    const faceOutline = [
      10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
      397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
      172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ];

    maskCtx.beginPath();
    const first = landmarks[faceOutline[0]];
    maskCtx.moveTo(first.x * width, first.y * height);
    for (let i = 1; i < faceOutline.length; i++) {
      const lm = landmarks[faceOutline[i]];
      maskCtx.lineTo(lm.x * width, lm.y * height);
    }
    maskCtx.closePath();
    maskCtx.fillStyle = 'white';
    maskCtx.fill();

    // Get only the face pixels
    const videoData = ctx.getImageData(0, 0, width, height);
    const maskData = maskCtx.getImageData(0, 0, width, height);

    let rSum = 0, gSum = 0, bSum = 0;
    let rSqSum = 0, gSqSum = 0, bSqSum = 0;
    let count = 0;

    for (let i = 0; i < videoData.data.length; i += 4) {
      if (maskData.data[i] > 128) { // Inside face region
        const r = videoData.data[i];
        const g = videoData.data[i + 1];
        const b = videoData.data[i + 2];

        rSum += r;
        gSum += g;
        bSum += b;

        rSqSum += r * r;
        gSqSum += g * g;
        bSqSum += b * b;

        count++;
      }
    }

    if (count === 0) {
      return { meanR: 128, meanG: 128, meanB: 128, stdR: 50, stdG: 50, stdB: 50 };
    }

    const meanR = rSum / count;
    const meanG = gSum / count;
    const meanB = bSum / count;

    const stdR = Math.sqrt(rSqSum / count - meanR * meanR);
    const stdG = Math.sqrt(gSqSum / count - meanG * meanG);
    const stdB = Math.sqrt(bSqSum / count - meanB * meanB);

    return { meanR, meanG, meanB, stdR, stdG, stdB };
  }

  // Apply color transfer from target stats to source texture
  applyColorTransfer(sourceCanvas, sourceStats, targetStats) {
    const width = sourceCanvas.width;
    const height = sourceCanvas.height;

    const outputCanvas = document.createElement('canvas');
    outputCanvas.width = width;
    outputCanvas.height = height;
    const ctx = outputCanvas.getContext('2d');
    ctx.drawImage(sourceCanvas, 0, 0);

    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
      const alpha = data[i + 3];
      if (alpha > 0) {
        // Normalize, scale by target stats, and denormalize
        let r = data[i];
        let g = data[i + 1];
        let b = data[i + 2];

        // Apply color transfer formula
        // new_pixel = (pixel - source_mean) * (target_std / source_std) + target_mean
        const stdRatioR = sourceStats.stdR > 0 ? targetStats.stdR / sourceStats.stdR : 1;
        const stdRatioG = sourceStats.stdG > 0 ? targetStats.stdG / sourceStats.stdG : 1;
        const stdRatioB = sourceStats.stdB > 0 ? targetStats.stdB / sourceStats.stdB : 1;

        r = (r - sourceStats.meanR) * stdRatioR + targetStats.meanR;
        g = (g - sourceStats.meanG) * stdRatioG + targetStats.meanG;
        b = (b - sourceStats.meanB) * stdRatioB + targetStats.meanB;

        // Clamp values
        data[i] = Math.max(0, Math.min(255, r));
        data[i + 1] = Math.max(0, Math.min(255, g));
        data[i + 2] = Math.max(0, Math.min(255, b));
      }
    }

    ctx.putImageData(imageData, 0, 0);
    return outputCanvas;
  }

  // Simple brightness matching
  matchBrightness(sourceCanvas, targetBrightness) {
    const ctx = sourceCanvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
    const data = imageData.data;

    // Calculate source brightness
    let sourceBrightness = 0;
    let count = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] > 128) {
        sourceBrightness += (data[i] + data[i + 1] + data[i + 2]) / 3;
        count++;
      }
    }
    sourceBrightness = count > 0 ? sourceBrightness / count : 128;

    // Calculate brightness adjustment
    const adjustment = targetBrightness - sourceBrightness;

    // Apply adjustment
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] > 0) {
        data[i] = Math.max(0, Math.min(255, data[i] + adjustment));
        data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + adjustment));
        data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + adjustment));
      }
    }

    ctx.putImageData(imageData, 0, 0);
    return sourceCanvas;
  }

  // Apply slight blur to edges for smoother blending
  featherEdges(canvas, featherRadius = 5) {
    const width = canvas.width;
    const height = canvas.height;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    // Create alpha channel copy
    const alphaChannel = new Float32Array(width * height);
    for (let i = 0; i < data.length; i += 4) {
      alphaChannel[i / 4] = data[i + 3];
    }

    // Apply Gaussian blur to alpha channel
    const blurredAlpha = this.gaussianBlur(alphaChannel, width, height, featherRadius);

    // Apply blurred alpha back
    for (let i = 0; i < data.length; i += 4) {
      const idx = i / 4;
      // Only modify edge pixels (where alpha differs from original)
      if (alphaChannel[idx] > 0 && alphaChannel[idx] < 255) {
        data[i + 3] = blurredAlpha[idx];
      }
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas;
  }

  // Simple box blur for alpha channel
  gaussianBlur(data, width, height, radius) {
    const output = new Float32Array(data.length);
    const size = radius * 2 + 1;
    const kernel = [];
    
    // Create Gaussian kernel
    let sum = 0;
    for (let i = -radius; i <= radius; i++) {
      const val = Math.exp(-(i * i) / (2 * radius * radius));
      kernel.push(val);
      sum += val;
    }
    for (let i = 0; i < kernel.length; i++) {
      kernel[i] /= sum;
    }

    // Horizontal pass
    const temp = new Float32Array(data.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let val = 0;
        for (let k = -radius; k <= radius; k++) {
          const sx = Math.max(0, Math.min(width - 1, x + k));
          val += data[y * width + sx] * kernel[k + radius];
        }
        temp[y * width + x] = val;
      }
    }

    // Vertical pass
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let val = 0;
        for (let k = -radius; k <= radius; k++) {
          const sy = Math.max(0, Math.min(height - 1, y + k));
          val += temp[sy * width + x] * kernel[k + radius];
        }
        output[y * width + x] = val;
      }
    }

    return output;
  }
}
