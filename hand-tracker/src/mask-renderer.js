// Mask Renderer - Draws face masks that follow facial landmarks

export class MaskRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.maskImage = null;
    this.maskLoaded = false;
    this.useProceduralMask = false;
    
    // Mask configuration (adjust based on your mask image)
    this.maskConfig = {
      // Reference width between eyes in the mask image (for scaling)
      referenceEyeDistance: 120,
      // Offset from face center to mask center (in mask image pixels)
      offsetX: 0,
      offsetY: -20,  // Move mask up slightly
      // Scale multiplier
      scaleMultiplier: 2.5
    };
  }

  async loadMask(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        this.maskImage = img;
        this.maskLoaded = true;
        console.log('Mask loaded:', img.width, 'x', img.height);
        resolve(img);
      };
      img.onerror = reject;
      img.src = url;
    });
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  // Draw mask using face transform
  drawMask(faceTransform) {
    if (!faceTransform) return;

    // Use procedural mask if no image loaded
    if (this.useProceduralMask || !this.maskLoaded) {
      this.drawProceduralMask(faceTransform);
      return;
    }

    const { center, angle, eyeDistance } = faceTransform;
    const config = this.maskConfig;

    // Calculate mask scale based on eye distance
    const maskScale = (eyeDistance / config.referenceEyeDistance) * config.scaleMultiplier;

    // Save context state
    this.ctx.save();

    // Move to face center
    this.ctx.translate(center.x, center.y);

    // Rotate to match face angle
    this.ctx.rotate(angle);

    // Scale the mask
    this.ctx.scale(maskScale, maskScale);

    // Draw the mask image centered
    const maskWidth = this.maskImage.width;
    const maskHeight = this.maskImage.height;
    
    this.ctx.drawImage(
      this.maskImage,
      -maskWidth / 2 + config.offsetX,
      -maskHeight / 2 + config.offsetY,
      maskWidth,
      maskHeight
    );

    // Restore context state
    this.ctx.restore();
  }

  // Draw a procedural "superhero" style mask
  drawProceduralMask(faceTransform) {
    const { leftEye, rightEye, nose, eyeDistance, angle, center } = faceTransform;

    this.ctx.save();
    this.ctx.translate(center.x, center.y);
    this.ctx.rotate(angle);

    const scale = eyeDistance / 80;
    this.ctx.scale(scale, scale);

    // Draw mask shape (covers eyes area)
    this.ctx.beginPath();
    
    // Left side
    this.ctx.moveTo(-120, -30);
    this.ctx.bezierCurveTo(-100, -60, -60, -50, -40, -30);
    
    // Left eye hole
    this.ctx.bezierCurveTo(-35, -20, -35, 10, -45, 15);
    this.ctx.bezierCurveTo(-55, 20, -80, 15, -90, 0);
    this.ctx.bezierCurveTo(-100, -15, -110, -20, -120, -30);
    
    // Nose bridge
    this.ctx.moveTo(-40, -30);
    this.ctx.bezierCurveTo(-20, -40, 20, -40, 40, -30);
    
    // Right side
    this.ctx.bezierCurveTo(60, -50, 100, -60, 120, -30);
    this.ctx.bezierCurveTo(110, -20, 100, -15, 90, 0);
    this.ctx.bezierCurveTo(80, 15, 55, 20, 45, 15);
    this.ctx.bezierCurveTo(35, 10, 35, -20, 40, -30);

    // Fill with gradient
    const gradient = this.ctx.createLinearGradient(-120, -50, 120, 20);
    gradient.addColorStop(0, 'rgba(139, 0, 0, 0.85)');
    gradient.addColorStop(0.5, 'rgba(180, 0, 0, 0.9)');
    gradient.addColorStop(1, 'rgba(139, 0, 0, 0.85)');
    
    this.ctx.fillStyle = gradient;
    this.ctx.fill();

    // Add border
    this.ctx.strokeStyle = 'rgba(255, 215, 0, 0.8)';
    this.ctx.lineWidth = 3;
    this.ctx.stroke();

    // Draw eye holes (cut outs)
    this.ctx.globalCompositeOperation = 'destination-out';
    
    // Left eye hole
    this.ctx.beginPath();
    this.ctx.ellipse(-55, -5, 25, 18, -0.2, 0, Math.PI * 2);
    this.ctx.fill();
    
    // Right eye hole  
    this.ctx.beginPath();
    this.ctx.ellipse(55, -5, 25, 18, 0.2, 0, Math.PI * 2);
    this.ctx.fill();

    this.ctx.globalCompositeOperation = 'source-over';

    // Add eye hole borders
    this.ctx.strokeStyle = 'rgba(255, 215, 0, 0.6)';
    this.ctx.lineWidth = 2;
    
    this.ctx.beginPath();
    this.ctx.ellipse(-55, -5, 25, 18, -0.2, 0, Math.PI * 2);
    this.ctx.stroke();
    
    this.ctx.beginPath();
    this.ctx.ellipse(55, -5, 25, 18, 0.2, 0, Math.PI * 2);
    this.ctx.stroke();

    this.ctx.restore();
  }

  // Draw debug visualization of landmarks
  drawLandmarks(landmarks, canvasWidth, canvasHeight) {
    if (!landmarks) return;

    this.ctx.fillStyle = 'rgba(0, 255, 136, 0.5)';

    for (const lm of landmarks) {
      const x = lm.x * canvasWidth;
      const y = lm.y * canvasHeight;
      
      this.ctx.beginPath();
      this.ctx.arc(x, y, 1, 0, Math.PI * 2);
      this.ctx.fill();
    }
  }

  // Draw mesh lines connecting landmarks
  drawMesh(landmarks, canvasWidth, canvasHeight) {
    if (!landmarks || landmarks.length < 468) return;

    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    this.ctx.strokeStyle = 'rgba(0, 217, 255, 0.3)';
    this.ctx.lineWidth = 0.5;

    // Draw face oval
    const silhouette = [
      10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
      397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
      172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ];

    this.ctx.beginPath();
    const first = toPixel(silhouette[0]);
    this.ctx.moveTo(first.x, first.y);
    for (let i = 1; i < silhouette.length; i++) {
      const p = toPixel(silhouette[i]);
      this.ctx.lineTo(p.x, p.y);
    }
    this.ctx.closePath();
    this.ctx.stroke();

    // Draw eyes
    this.ctx.strokeStyle = 'rgba(0, 255, 136, 0.5)';
    const leftEye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
    const rightEye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];

    for (const eye of [leftEye, rightEye]) {
      this.ctx.beginPath();
      const start = toPixel(eye[0]);
      this.ctx.moveTo(start.x, start.y);
      for (let i = 1; i < eye.length; i++) {
        const p = toPixel(eye[i]);
        this.ctx.lineTo(p.x, p.y);
      }
      this.ctx.closePath();
      this.ctx.stroke();
    }

    // Draw lips
    this.ctx.strokeStyle = 'rgba(255, 100, 150, 0.5)';
    const lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
    
    this.ctx.beginPath();
    const lipStart = toPixel(lips[0]);
    this.ctx.moveTo(lipStart.x, lipStart.y);
    for (let i = 1; i < lips.length; i++) {
      const p = toPixel(lips[i]);
      this.ctx.lineTo(p.x, p.y);
    }
    this.ctx.closePath();
    this.ctx.stroke();
  }

  // Draw key reference points
  drawKeyPoints(faceTransform) {
    if (!faceTransform) return;

    const { leftEye, rightEye, nose, forehead, chin, center } = faceTransform;

    this.ctx.fillStyle = '#00ff88';
    
    // Draw key points
    for (const point of [leftEye, rightEye, nose, forehead, chin]) {
      this.ctx.beginPath();
      this.ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
      this.ctx.fill();
    }

    // Draw center cross
    this.ctx.strokeStyle = '#ff0000';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(center.x - 10, center.y);
    this.ctx.lineTo(center.x + 10, center.y);
    this.ctx.moveTo(center.x, center.y - 10);
    this.ctx.lineTo(center.x, center.y + 10);
    this.ctx.stroke();
  }

  // Draw expression-reactive effects
  drawExpressionEffects(landmarks, canvasWidth, canvasHeight, expressions) {
    if (!landmarks || !expressions) return;

    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    this.ctx.save();

    // Smile effect - draw sparkles near mouth corners when smiling
    if (expressions.smileAmount > 0.3) {
      const leftMouthCorner = toPixel(61);
      const rightMouthCorner = toPixel(291);
      const intensity = expressions.smileAmount;
      
      this.ctx.fillStyle = `rgba(255, 215, 0, ${intensity * 0.8})`;
      
      // Draw sparkles
      for (const corner of [leftMouthCorner, rightMouthCorner]) {
        const sparkleSize = 4 + intensity * 6;
        this.drawSparkle(corner.x + (corner === leftMouthCorner ? -15 : 15), corner.y - 10, sparkleSize);
      }
    }

    // Mouth open effect - draw "aura" inside mouth
    if (expressions.mouthOpenAmount > 0.4) {
      const mouthCenter = toPixel(13); // Inner lip top center
      const mouthBottom = toPixel(14); // Inner lip bottom center
      const openAmount = expressions.mouthOpenAmount;
      
      const mouthOpenSize = Math.abs(mouthBottom.y - mouthCenter.y);
      
      // Draw gradient inside mouth
      const gradient = this.ctx.createRadialGradient(
        mouthCenter.x, (mouthCenter.y + mouthBottom.y) / 2, 0,
        mouthCenter.x, (mouthCenter.y + mouthBottom.y) / 2, mouthOpenSize * 2
      );
      gradient.addColorStop(0, `rgba(255, 100, 50, ${openAmount * 0.6})`);
      gradient.addColorStop(0.5, `rgba(255, 50, 0, ${openAmount * 0.3})`);
      gradient.addColorStop(1, 'rgba(255, 0, 0, 0)');
      
      this.ctx.fillStyle = gradient;
      this.ctx.beginPath();
      this.ctx.ellipse(
        mouthCenter.x, 
        (mouthCenter.y + mouthBottom.y) / 2, 
        mouthOpenSize * 1.5, 
        mouthOpenSize, 
        0, 0, Math.PI * 2
      );
      this.ctx.fill();
    }

    // Brow raise effect - draw emphasis lines above eyebrows
    if (expressions.browRaiseAmount > 0.3) {
      const leftBrow = toPixel(70);  // Left eyebrow
      const rightBrow = toPixel(300); // Right eyebrow
      const intensity = expressions.browRaiseAmount;
      
      this.ctx.strokeStyle = `rgba(0, 200, 255, ${intensity * 0.6})`;
      this.ctx.lineWidth = 2;
      this.ctx.lineCap = 'round';
      
      for (const brow of [leftBrow, rightBrow]) {
        this.ctx.beginPath();
        this.ctx.moveTo(brow.x - 20, brow.y - 15 - intensity * 10);
        this.ctx.lineTo(brow.x + 20, brow.y - 15 - intensity * 10);
        this.ctx.stroke();
      }
    }

    // Eye closed effect - draw "Z" for sleeping
    if (expressions.bothEyesClosed) {
      const forehead = toPixel(10);
      
      this.ctx.font = 'bold 24px sans-serif';
      this.ctx.fillStyle = 'rgba(100, 150, 255, 0.8)';
      this.ctx.fillText('z', forehead.x + 30, forehead.y - 20);
      this.ctx.font = 'bold 18px sans-serif';
      this.ctx.fillText('z', forehead.x + 50, forehead.y - 35);
      this.ctx.font = 'bold 14px sans-serif';
      this.ctx.fillText('z', forehead.x + 65, forehead.y - 45);
    }

    this.ctx.restore();
  }

  // Helper to draw a sparkle/star shape
  drawSparkle(x, y, size) {
    this.ctx.save();
    this.ctx.translate(x, y);
    
    this.ctx.beginPath();
    for (let i = 0; i < 4; i++) {
      const angle = (i * Math.PI) / 2;
      const innerAngle = angle + Math.PI / 4;
      
      this.ctx.lineTo(
        Math.cos(angle) * size,
        Math.sin(angle) * size
      );
      this.ctx.lineTo(
        Math.cos(innerAngle) * size * 0.4,
        Math.sin(innerAngle) * size * 0.4
      );
    }
    this.ctx.closePath();
    this.ctx.fill();
    
    this.ctx.restore();
  }
}
