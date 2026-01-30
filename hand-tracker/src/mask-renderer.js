// Mask Renderer - Draws face masks that follow facial landmarks

// MediaPipe Face Mesh triangulation - 468 vertices forming triangles
// Each group of 3 numbers represents a triangle (vertex indices)
const FACE_MESH_TRIANGLES = [
  127, 34, 139, 11, 0, 37, 232, 231, 120, 72, 37, 39, 128, 121, 47, 232, 121, 128,
  104, 69, 67, 175, 171, 148, 157, 154, 155, 118, 50, 101, 73, 39, 40, 9, 151, 108,
  48, 115, 131, 194, 204, 211, 74, 40, 185, 80, 42, 183, 40, 92, 186, 230, 229, 118,
  202, 212, 214, 83, 18, 17, 76, 61, 146, 160, 29, 30, 56, 157, 173, 106, 204, 194,
  135, 214, 192, 203, 165, 98, 21, 71, 68, 51, 45, 4, 144, 24, 23, 77, 146, 91,
  205, 50, 187, 201, 200, 18, 91, 106, 182, 90, 91, 181, 85, 84, 17, 206, 203, 36,
  148, 171, 140, 92, 40, 39, 193, 189, 244, 159, 158, 28, 247, 246, 161, 236, 3, 196,
  54, 68, 104, 193, 168, 8, 117, 228, 31, 189, 193, 55, 98, 97, 99, 126, 47, 100,
  166, 79, 218, 155, 154, 26, 209, 49, 131, 135, 136, 150, 47, 126, 217, 223, 52, 53,
  45, 51, 134, 211, 170, 140, 67, 69, 108, 43, 106, 91, 230, 119, 120, 226, 130, 247,
  63, 53, 52, 238, 20, 242, 46, 70, 156, 78, 62, 96, 46, 53, 63, 143, 34, 227,
  173, 155, 133, 123, 117, 111, 44, 125, 19, 236, 134, 51, 216, 206, 205, 154, 153, 22,
  39, 37, 167, 200, 201, 208, 36, 142, 100, 57, 212, 202, 20, 60, 99, 28, 158, 157,
  35, 226, 113, 160, 159, 27, 204, 202, 210, 113, 225, 46, 43, 202, 204, 62, 76, 77,
  137, 123, 116, 41, 38, 72, 203, 129, 142, 64, 98, 240, 49, 102, 64, 41, 73, 74,
  212, 216, 207, 42, 74, 184, 169, 170, 211, 170, 149, 176, 105, 66, 69, 122, 6, 168,
  123, 147, 187, 96, 77, 90, 65, 55, 107, 89, 90, 180, 101, 100, 120, 63, 105, 104,
  93, 137, 227, 15, 86, 85, 129, 102, 49, 14, 87, 86, 55, 8, 9, 100, 47, 121,
  145, 23, 22, 88, 89, 179, 6, 122, 196, 88, 95, 96, 138, 172, 136, 215, 58, 172,
  115, 48, 219, 42, 80, 81, 195, 3, 51, 43, 146, 61, 171, 175, 199, 81, 82, 38,
  53, 46, 225, 144, 163, 110, 52, 65, 66, 229, 228, 117, 34, 127, 234, 107, 108, 69,
  109, 108, 151, 48, 64, 235, 62, 78, 191, 129, 209, 126, 111, 35, 143, 117, 123, 50,
  222, 65, 52, 19, 125, 141, 221, 55, 65, 3, 195, 197, 25, 7, 33, 220, 237, 44,
  70, 71, 139, 122, 193, 245, 247, 130, 33, 71, 21, 162, 153, 158, 159, 170, 169, 150,
  188, 174, 196, 216, 186, 92, 144, 160, 161, 2, 97, 167, 141, 125, 241, 164, 167, 37,
  72, 38, 12, 38, 82, 13, 63, 68, 71, 226, 35, 111, 101, 50, 205, 206, 92, 165,
  209, 198, 217, 165, 167, 97, 220, 115, 218, 133, 112, 243, 239, 238, 241, 214, 135, 169,
  190, 173, 133, 171, 208, 32, 125, 44, 237, 86, 87, 178, 85, 86, 179, 84, 85, 180,
  83, 84, 181, 201, 83, 182, 137, 93, 132, 76, 62, 183, 61, 76, 184, 57, 61, 185,
  212, 57, 186, 214, 207, 187, 34, 143, 156, 79, 239, 237, 123, 137, 177, 44, 1, 4,
  201, 194, 32, 64, 102, 129, 213, 215, 138, 59, 166, 219, 242, 99, 97, 2, 94, 141,
  75, 59, 235, 24, 110, 228, 25, 130, 226, 23, 24, 229, 22, 23, 230, 26, 22, 231,
  112, 26, 232, 189, 190, 243, 221, 56, 190, 28, 56, 221, 27, 28, 222, 29, 27, 223,
  30, 29, 224, 247, 30, 225, 238, 79, 20, 166, 59, 75, 60, 75, 240, 147, 177, 215,
  20, 79, 166, 187, 147, 213, 112, 233, 244, 128, 121, 232, 245, 122, 188, 188, 114, 174,
  134, 131, 220, 174, 217, 236, 236, 198, 134, 215, 177, 58, 156, 143, 124, 25, 110, 7,
  31, 228, 25, 264, 356, 368, 0, 11, 267, 451, 452, 349, 267, 302, 269, 350, 357, 277,
  350, 452, 357, 299, 333, 297, 396, 175, 377, 381, 384, 382, 280, 347, 330, 269, 303, 270,
  151, 9, 337, 344, 278, 360, 262, 351, 343, 333, 299, 334, 168, 417, 351, 352, 280, 411,
  325, 319, 320, 295, 296, 336, 166, 79, 218, 384, 381, 256, 252, 253, 426, 391, 393, 267,
  107, 55, 65, 423, 364, 379, 416, 417, 402, 361, 360, 291, 407, 408, 306, 415, 310, 311,
  310, 415, 407, 313, 314, 17, 306, 408, 307, 418, 408, 315, 315, 16, 14, 314, 313, 12,
  312, 268, 13, 298, 293, 301, 265, 446, 340, 280, 330, 425, 322, 426, 391, 420, 424, 356,
  364, 379, 394, 379, 378, 395, 378, 377, 394, 293, 388, 301, 265, 340, 261, 388, 466, 249,
  390, 373, 368, 255, 339, 254, 448, 261, 340, 390, 466, 388, 342, 448, 449, 438, 309, 392,
  289, 455, 439, 250, 309, 455, 290, 305, 392, 305, 250, 290, 328, 327, 326, 261, 446, 448,
  367, 302, 301, 359, 263, 464, 324, 325, 391, 303, 271, 304, 436, 432, 427, 304, 272, 408,
  395, 394, 431, 378, 395, 400, 296, 334, 299, 6, 351, 168, 376, 352, 411, 307, 325, 320,
  285, 295, 336, 320, 319, 404, 329, 330, 349, 334, 293, 333, 366, 447, 346, 318, 319, 319,
  277, 440, 278, 439, 455, 289, 344, 278, 282, 388, 293, 334, 286, 258, 259, 447, 286, 346,
  340, 446, 265, 276, 353, 282, 424, 442, 441, 353, 276, 283, 265, 357, 452, 453, 341, 464,
  250, 458, 462, 276, 356, 353, 282, 353, 331, 283, 276, 282, 354, 370, 373, 295, 384, 380,
  381, 295, 380, 374, 380, 295, 326, 328, 304, 292, 325, 307, 366, 447, 345, 319, 318, 403,
  305, 290, 250, 374, 375, 404, 321, 408, 272, 272, 318, 304, 403, 408, 316, 318, 272, 303,
  307, 270, 408, 408, 270, 409, 305, 270, 303, 270, 305, 408, 269, 270, 305, 400, 377, 152,
  368, 264, 383, 423, 395, 400, 397, 435, 391, 344, 438, 390, 326, 327, 463, 463, 327, 460,
  399, 326, 459, 412, 355, 462, 367, 364, 379, 367, 394, 395, 359, 464, 263, 367, 379, 380,
  359, 474, 263, 393, 380, 391, 255, 339, 249, 390, 249, 466, 339, 255, 249, 358, 461, 462,
  343, 357, 465, 412, 343, 465, 437, 422, 343, 343, 422, 457, 253, 252, 381, 256, 253, 381,
  349, 451, 452, 340, 261, 448, 297, 333, 298, 374, 375, 406, 434, 430, 431, 394, 378, 395,
  395, 431, 430, 396, 377, 400, 411, 376, 284, 262, 463, 343, 262, 331, 463, 343, 463, 331,
  369, 299, 297, 345, 350, 277, 442, 434, 450, 450, 467, 445, 362, 398, 359, 399, 326, 463,
  474, 359, 362, 395, 423, 430, 398, 359, 263, 455, 309, 438, 309, 392, 250, 376, 433, 284,
  250, 392, 458, 435, 432, 416, 435, 416, 389, 393, 391, 268, 285, 417, 8, 351, 6, 168,
  412, 465, 343, 282, 283, 331, 432, 436, 416, 460, 327, 458, 458, 327, 326, 459, 326, 458,
  411, 280, 376, 284, 433, 411, 285, 336, 417, 6, 122, 168, 364, 367, 379, 406, 378, 433,
  395, 430, 434, 434, 442, 430
];

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

  // Draw triangle mesh over the face
  drawTriangleMesh(landmarks, canvasWidth, canvasHeight, options = {}) {
    if (!landmarks || landmarks.length < 468) return;

    const {
      lineWidth = 0.5,
      showVertices = false,
      vertexRadius = 1.5,
      useDepth = false,
      showContours = true,
      showTriangles = true,
      strokeColor = '#00ffff',
      fillColor = '#ffffff',
      fillOpacity = 0.05,
      // Contour options
      contourWidth = 1.5,
      contourColor = '#00ff88',
      showEyes = true,
      showEyebrows = true,
      showLips = true,
      showNose = true,
      showFaceOval = true
    } = options;

    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight,
      z: landmarks[idx].z || 0
    });

    this.ctx.save();

    // Calculate Z range for depth normalization
    let minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < Math.min(landmarks.length, 468); i++) {
      const z = landmarks[i].z || 0;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }
    const zRange = maxZ - minZ || 0.1;

    // Get depth-based color with more vivid gradient
    const getDepthColor = (z, alpha = 0.5) => {
      const t = (z - minZ) / zRange; // 0 = closest, 1 = furthest
      
      // Vivid gradient: Yellow/Orange (close) -> Cyan -> Purple (far)
      let r, g, b;
      if (t < 0.5) {
        // Close: warm colors (yellow to cyan)
        const t2 = t * 2;
        r = Math.round(255 * (1 - t2));
        g = Math.round(220 + 35 * t2);
        b = Math.round(50 + 205 * t2);
      } else {
        // Far: cool colors (cyan to purple)
        const t2 = (t - 0.5) * 2;
        r = Math.round(0 + 150 * t2);
        g = Math.round(255 * (1 - t2 * 0.6));
        b = 255;
      }
      
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    };

    // Collect all triangles, skip those with out-of-bounds vertices
    const triangles = [];
    
    for (let i = 0; i < FACE_MESH_TRIANGLES.length; i += 3) {
      const idx0 = FACE_MESH_TRIANGLES[i];
      const idx1 = FACE_MESH_TRIANGLES[i + 1];
      const idx2 = FACE_MESH_TRIANGLES[i + 2];

      if (idx0 >= landmarks.length || idx1 >= landmarks.length || idx2 >= landmarks.length) {
        continue;
      }

      const p0 = toPixel(idx0);
      const p1 = toPixel(idx1);
      const p2 = toPixel(idx2);
      
      // Calculate average Z for depth coloring
      const avgZ = (p0.z + p1.z + p2.z) / 3;
      
      triangles.push({ p0, p1, p2, avgZ });
    }

    // Don't sort - draw in original order to maintain symmetry

    // Draw all triangles uniformly
    this.ctx.lineWidth = lineWidth;
    
    // Parse fill color for opacity
    const hexToRgb = (hex) => {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
      } : { r: 255, g: 255, b: 255 };
    };
    
    const fillRgb = hexToRgb(fillColor);
    const strokeRgb = hexToRgb(strokeColor);
    
    if (showTriangles) {
      for (const tri of triangles) {
        const { p0, p1, p2, avgZ } = tri;
        
        this.ctx.beginPath();
        this.ctx.moveTo(p0.x, p0.y);
        this.ctx.lineTo(p1.x, p1.y);
        this.ctx.lineTo(p2.x, p2.y);
        this.ctx.closePath();

        if (useDepth) {
          this.ctx.fillStyle = getDepthColor(avgZ, fillOpacity * 2);
          this.ctx.strokeStyle = getDepthColor(avgZ, 0.35);
          if (fillOpacity > 0) this.ctx.fill();
        } else {
          this.ctx.fillStyle = `rgba(${fillRgb.r}, ${fillRgb.g}, ${fillRgb.b}, ${fillOpacity})`;
          this.ctx.strokeStyle = `rgba(${strokeRgb.r}, ${strokeRgb.g}, ${strokeRgb.b}, 0.4)`;
          if (fillOpacity > 0) this.ctx.fill();
        }
        this.ctx.stroke();
      }
    }

    // Draw facial contours for definition
    if (showContours) {
      this.drawFaceContours(landmarks, canvasWidth, canvasHeight, {
        getDepthColor: useDepth ? getDepthColor : null,
        contourWidth,
        contourColor,
        showEyes,
        showEyebrows,
        showLips,
        showNose,
        showFaceOval
      });
    }

    // Draw vertices
    if (showVertices && vertexRadius > 0) {
      for (let i = 0; i < Math.min(landmarks.length, 468); i++) {
        const p = toPixel(i);
        
        if (useDepth) {
          this.ctx.fillStyle = getDepthColor(p.z, 0.8);
        } else {
          this.ctx.fillStyle = `rgba(${strokeRgb.r}, ${strokeRgb.g}, ${strokeRgb.b}, 0.7)`;
        }
        
        this.ctx.beginPath();
        this.ctx.arc(p.x, p.y, vertexRadius, 0, Math.PI * 2);
        this.ctx.fill();
      }
    }

    this.ctx.restore();
  }

  // Draw facial contours (eyes, lips, face oval, eyebrows)
  drawFaceContours(landmarks, canvasWidth, canvasHeight, options = {}) {
    const {
      getDepthColor = null,
      contourWidth = 1.5,
      contourColor = '#00ff88',
      showEyes = true,
      showEyebrows = true,
      showLips = true,
      showNose = true,
      showFaceOval = true
    } = options;
    
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight,
      z: landmarks[idx].z || 0
    });

    // Contour definitions with visibility flags
    const contours = {
      faceOval: { indices: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10], visible: showFaceOval },
      leftEye: { indices: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33], visible: showEyes },
      rightEye: { indices: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362], visible: showEyes },
      leftEyebrow: { indices: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46], visible: showEyebrows },
      rightEyebrow: { indices: [300, 293, 334, 296, 336, 285, 295, 282, 283, 276], visible: showEyebrows },
      lipsOuter: { indices: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61], visible: showLips },
      lipsInner: { indices: [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78], visible: showLips },
      noseBridge: { indices: [168, 6, 197, 195, 5, 4, 1], visible: showNose },
      noseBottom: { indices: [129, 98, 97, 2, 326, 327, 358], visible: showNose }
    };

    // Parse contour color
    const hexToRgb = (hex) => {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
      } : { r: 0, g: 255, b: 136 };
    };
    const rgb = hexToRgb(contourColor);

    const defaultColors = {
      faceOval: `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.4)`,
      leftEye: `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.7)`,
      rightEye: `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.7)`,
      leftEyebrow: `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.5)`,
      rightEyebrow: `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.5)`,
      lipsOuter: `rgba(255, 120, 150, 0.6)`,
      lipsInner: `rgba(255, 80, 120, 0.5)`,
      noseBridge: `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.4)`,
      noseBottom: `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.4)`
    };

    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';

    for (const [name, contour] of Object.entries(contours)) {
      if (!contour.visible || contour.indices.length < 2) continue;

      this.ctx.beginPath();
      const first = toPixel(contour.indices[0]);
      this.ctx.moveTo(first.x, first.y);

      for (let i = 1; i < contour.indices.length; i++) {
        if (contour.indices[i] >= landmarks.length) continue;
        const p = toPixel(contour.indices[i]);
        this.ctx.lineTo(p.x, p.y);
      }

      // Use depth color if available, otherwise use preset color
      if (getDepthColor) {
        const avgZ = contour.indices.reduce((sum, idx) => sum + (landmarks[idx]?.z || 0), 0) / contour.indices.length;
        this.ctx.strokeStyle = getDepthColor(avgZ, 0.7);
      } else {
        this.ctx.strokeStyle = defaultColors[name];
      }
      
      this.ctx.lineWidth = name === 'faceOval' ? contourWidth * 1.3 : contourWidth;
      this.ctx.stroke();
    }
  }
  
  // Draw landmark index numbers
  drawLandmarkIndices(landmarks, canvasWidth, canvasHeight) {
    if (!landmarks || landmarks.length === 0) return;
    
    this.ctx.font = '8px monospace';
    this.ctx.fillStyle = 'rgba(255, 255, 0, 0.7)';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    
    // Only draw every 10th landmark to avoid clutter
    for (let i = 0; i < Math.min(landmarks.length, 468); i += 10) {
      const x = landmarks[i].x * canvasWidth;
      const y = landmarks[i].y * canvasHeight;
      this.ctx.fillText(i.toString(), x, y);
    }
  }
  
  // Draw FPS counter
  drawFPS(fps) {
    this.ctx.save();
    this.ctx.font = 'bold 14px monospace';
    this.ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
    this.ctx.textAlign = 'left';
    this.ctx.textBaseline = 'top';
    this.ctx.fillText(`${fps} FPS`, 10, 10);
    this.ctx.restore();
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
