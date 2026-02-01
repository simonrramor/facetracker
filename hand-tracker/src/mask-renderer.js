// Mask Renderer - Draws face masks that follow facial landmarks

import { adaptiveTriangulation, FACIAL_REGIONS } from './adaptive-triangulation.js';
import { HAND_CONNECTIONS, HAND_LANDMARKS, HandTracker } from './hand-tracker.js';

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

  // Draw custom procedural mask using landmarks and design settings
  // Now with 3D depth shading based on face Z coordinates
  drawCustomMask(landmarks, canvasWidth, canvasHeight, design = {}) {
    if (!landmarks || landmarks.length < 468) return;

    const {
      shape = 'classic',
      coverage = 'upper',
      primaryColor = '#1a1a2e',
      secondaryColor = '#16213e',
      borderColor = '#e94560',
      useGradient = true,
      borderWidth = 3,
      borderGlow = true,
      glowIntensity = 0.5,
      eyeHoleShape = 'pointed',
      eyeHoleSize = 1.0,
      eyeHoleBorder = true,
      opacity = 0.95
    } = design;

    // Helper to get pixel position with Z from landmark index
    const toPixel3D = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight,
      z: landmarks[idx].z || 0
    });

    // Calculate Z range for depth normalization
    let minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < Math.min(landmarks.length, 468); i++) {
      const z = landmarks[i].z || 0;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }
    const zRange = maxZ - minZ || 0.1;

    // Parse colors to RGB
    const hexToRgb = (hex) => {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
      } : { r: 26, g: 26, b: 46 };
    };

    const primaryRgb = hexToRgb(primaryColor);
    const secondaryRgb = hexToRgb(secondaryColor);

    // Key landmark positions
    const forehead = toPixel3D(10);
    const leftTemple = toPixel3D(127);
    const rightTemple = toPixel3D(356);
    const noseTip = toPixel3D(1);
    const noseBridge = toPixel3D(6);
    const leftEyeOuter = toPixel3D(33);
    const rightEyeOuter = toPixel3D(263);

    this.ctx.save();
    this.ctx.globalAlpha = opacity;

    // Get mask region landmarks based on shape
    const maskLandmarks = this.getMaskLandmarks(shape, coverage);
    
    // Draw depth-shaded triangulated mask
    this.drawDepthShadedMask(landmarks, canvasWidth, canvasHeight, maskLandmarks, {
      primaryRgb,
      secondaryRgb,
      minZ,
      zRange,
      useGradient
    });

    // Draw border outline
    if (borderWidth > 0) {
      this.ctx.beginPath();
      if (shape === 'classic' || shape === 'angular') {
        this.drawUpperMaskShape(landmarks, canvasWidth, canvasHeight, shape);
      } else if (shape === 'masquerade') {
        this.drawMasqueradeMaskShape(landmarks, canvasWidth, canvasHeight);
      } else if (shape === 'ninja') {
        this.drawNinjaMaskShape(landmarks, canvasWidth, canvasHeight);
      } else if (shape === 'phantom') {
        this.drawPhantomMaskShape(landmarks, canvasWidth, canvasHeight);
      }

      if (borderGlow && glowIntensity > 0) {
        this.ctx.shadowColor = borderColor;
        this.ctx.shadowBlur = 15 * glowIntensity;
      }
      this.ctx.strokeStyle = borderColor;
      this.ctx.lineWidth = borderWidth;
      this.ctx.stroke();
      this.ctx.shadowBlur = 0;
    }

    // Cut out eye holes (not for ninja mask)
    if (shape !== 'ninja' && eyeHoleSize > 0) {
      this.ctx.globalCompositeOperation = 'destination-out';
      this.drawEyeHoles(landmarks, canvasWidth, canvasHeight, eyeHoleShape, eyeHoleSize);
      this.ctx.globalCompositeOperation = 'source-over';

      // Draw eye hole borders
      if (eyeHoleBorder && borderWidth > 0) {
        this.ctx.strokeStyle = borderColor;
        this.ctx.lineWidth = borderWidth * 0.7;
        if (borderGlow && glowIntensity > 0) {
          this.ctx.shadowColor = borderColor;
          this.ctx.shadowBlur = 10 * glowIntensity;
        }
        this.drawEyeHoleBorders(landmarks, canvasWidth, canvasHeight, eyeHoleShape, eyeHoleSize);
        this.ctx.shadowBlur = 0;
      }
    }

    this.ctx.restore();
  }

  // Get landmark indices for mask region based on shape
  getMaskLandmarks(shape, coverage) {
    // Upper face landmarks (forehead, eyes, nose bridge area)
    const upperFace = [
      10, 338, 297, 332, 284, 251, 389, 356, // right side upper
      127, 162, 21, 54, 103, 67, 109, // left side upper
      151, 108, 69, 104, 68, 71, 139, // forehead
      336, 296, 334, 293, 300, // right eyebrow area
      70, 63, 105, 66, 107, 55, 65, 52, 53, 46, // left eyebrow area
      168, 6, 197, 195, 5, 4, // nose bridge
      122, 193, 245, 188, 174, 236, 198, 209, 129, // around eyes
      351, 417, 465, 412, 399, 456, 420, 429, 358 // around eyes right
    ];

    // Lower face landmarks
    const lowerFace = [
      152, 148, 176, 149, 150, 136, 172, 58, 132, // chin and jaw left
      397, 365, 379, 378, 400, 377, // jaw right
      1, 2, 98, 97, 326, 327, // nose bottom
      61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291 // lips area
    ];

    // Phantom mask (left half)
    const phantomFace = [
      10, 109, 67, 103, 54, 21, 162, 127, // left upper
      234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, // left jaw
      70, 63, 105, 66, 107, 55, 65, 52, 53, 46, // left eyebrow
      168, 6, 197, 195, 5, 4, 1, // nose center
      33, 7, 163, 144, 145, 153, 154, 155, 133 // left eye area
    ];

    if (shape === 'ninja') return lowerFace;
    if (shape === 'phantom') return phantomFace;
    return upperFace;
  }

  // Draw depth-shaded mask using triangulated face mesh
  drawDepthShadedMask(landmarks, canvasWidth, canvasHeight, maskLandmarks, colorOptions) {
    const { primaryRgb, secondaryRgb, minZ, zRange, useGradient } = colorOptions;

    const toPixel3D = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight,
      z: landmarks[idx].z || 0
    });

    // Create a set of mask landmark indices for fast lookup
    const maskSet = new Set(maskLandmarks);

    // Filter triangles to only those with all vertices in mask region
    const maskTriangles = [];
    for (let i = 0; i < FACE_MESH_TRIANGLES.length; i += 3) {
      const idx0 = FACE_MESH_TRIANGLES[i];
      const idx1 = FACE_MESH_TRIANGLES[i + 1];
      const idx2 = FACE_MESH_TRIANGLES[i + 2];

      // Check if at least 2 vertices are in the mask region
      const inMask = [idx0, idx1, idx2].filter(idx => maskSet.has(idx)).length;
      if (inMask >= 2 && idx0 < landmarks.length && idx1 < landmarks.length && idx2 < landmarks.length) {
        const p0 = toPixel3D(idx0);
        const p1 = toPixel3D(idx1);
        const p2 = toPixel3D(idx2);
        const avgZ = (p0.z + p1.z + p2.z) / 3;
        maskTriangles.push({ p0, p1, p2, avgZ });
      }
    }

    // Sort by Z (furthest first - painter's algorithm)
    maskTriangles.sort((a, b) => b.avgZ - a.avgZ);

    // Draw each triangle with depth-based shading
    for (const tri of maskTriangles) {
      const { p0, p1, p2, avgZ } = tri;

      // Calculate depth factor (0 = far/dark, 1 = close/bright)
      const depthFactor = 1 - ((avgZ - minZ) / zRange);
      
      // Apply lighting - closer surfaces are brighter
      const lightIntensity = 0.4 + depthFactor * 0.6; // Range 0.4 to 1.0
      
      // Blend primary and secondary colors based on depth
      let r, g, b;
      if (useGradient) {
        const blend = depthFactor;
        r = Math.round(primaryRgb.r * (1 - blend) + secondaryRgb.r * blend);
        g = Math.round(primaryRgb.g * (1 - blend) + secondaryRgb.g * blend);
        b = Math.round(primaryRgb.b * (1 - blend) + secondaryRgb.b * blend);
      } else {
        r = primaryRgb.r;
        g = primaryRgb.g;
        b = primaryRgb.b;
      }

      // Apply lighting
      r = Math.round(r * lightIntensity);
      g = Math.round(g * lightIntensity);
      b = Math.round(b * lightIntensity);

      this.ctx.beginPath();
      this.ctx.moveTo(p0.x, p0.y);
      this.ctx.lineTo(p1.x, p1.y);
      this.ctx.lineTo(p2.x, p2.y);
      this.ctx.closePath();

      this.ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      this.ctx.fill();
    }
  }

  // Draw upper face mask shape (classic or angular)
  drawUpperMaskShape(landmarks, canvasWidth, canvasHeight, style) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    const forehead = toPixel(10);
    const leftTemple = toPixel(127);
    const rightTemple = toPixel(356);
    const leftCheek = toPixel(234);
    const rightCheek = toPixel(454);
    const noseBridge = toPixel(6);
    const noseTip = toPixel(1);
    const leftEyebrowOuter = toPixel(46);
    const rightEyebrowOuter = toPixel(276);
    const leftEyebrowInner = toPixel(55);
    const rightEyebrowInner = toPixel(285);

    if (style === 'angular') {
      // Sharp angular Batman-style mask
      this.ctx.moveTo(leftTemple.x - 10, leftTemple.y);
      this.ctx.lineTo(leftEyebrowOuter.x, forehead.y - 20);
      this.ctx.lineTo(forehead.x, forehead.y - 30);
      this.ctx.lineTo(rightEyebrowOuter.x, forehead.y - 20);
      this.ctx.lineTo(rightTemple.x + 10, rightTemple.y);
      this.ctx.lineTo(rightCheek.x, rightCheek.y - 10);
      this.ctx.lineTo(noseTip.x, noseTip.y - 20);
      this.ctx.lineTo(leftCheek.x, leftCheek.y - 10);
      this.ctx.closePath();
    } else {
      // Smooth classic mask
      this.ctx.moveTo(leftTemple.x - 5, leftTemple.y);
      this.ctx.quadraticCurveTo(leftEyebrowOuter.x, forehead.y - 15, forehead.x, forehead.y - 20);
      this.ctx.quadraticCurveTo(rightEyebrowOuter.x, forehead.y - 15, rightTemple.x + 5, rightTemple.y);
      this.ctx.quadraticCurveTo(rightCheek.x + 5, rightCheek.y, noseTip.x, noseTip.y - 15);
      this.ctx.quadraticCurveTo(leftCheek.x - 5, leftCheek.y, leftTemple.x - 5, leftTemple.y);
    }
  }

  // Draw masquerade mask shape (ornate with decorative edges)
  drawMasqueradeMaskShape(landmarks, canvasWidth, canvasHeight) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    const forehead = toPixel(10);
    const leftTemple = toPixel(127);
    const rightTemple = toPixel(356);
    const leftCheek = toPixel(234);
    const rightCheek = toPixel(454);
    const noseTip = toPixel(1);
    const leftEyebrowOuter = toPixel(46);
    const rightEyebrowOuter = toPixel(276);

    // Ornate shape with pointed edges
    this.ctx.moveTo(leftTemple.x - 20, leftTemple.y - 10);
    
    // Left decorative edge
    this.ctx.quadraticCurveTo(leftTemple.x - 30, forehead.y - 20, leftEyebrowOuter.x - 10, forehead.y - 35);
    this.ctx.quadraticCurveTo(forehead.x - 30, forehead.y - 45, forehead.x, forehead.y - 40);
    this.ctx.quadraticCurveTo(forehead.x + 30, forehead.y - 45, rightEyebrowOuter.x + 10, forehead.y - 35);
    this.ctx.quadraticCurveTo(rightTemple.x + 30, forehead.y - 20, rightTemple.x + 20, rightTemple.y - 10);
    
    // Right side
    this.ctx.quadraticCurveTo(rightCheek.x + 10, rightCheek.y - 5, noseTip.x, noseTip.y - 10);
    
    // Left side
    this.ctx.quadraticCurveTo(leftCheek.x - 10, leftCheek.y - 5, leftTemple.x - 20, leftTemple.y - 10);
  }

  // Draw ninja mask shape (lower face covering)
  drawNinjaMaskShape(landmarks, canvasWidth, canvasHeight) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    const noseTip = toPixel(1);
    const chin = toPixel(152);
    const leftJaw = toPixel(172);
    const rightJaw = toPixel(397);
    const leftCheek = toPixel(234);
    const rightCheek = toPixel(454);
    const leftEar = toPixel(234);
    const rightEar = toPixel(454);

    // Cover from nose down
    this.ctx.moveTo(leftCheek.x - 10, noseTip.y - 10);
    this.ctx.quadraticCurveTo(noseTip.x, noseTip.y - 20, rightCheek.x + 10, noseTip.y - 10);
    this.ctx.lineTo(rightJaw.x + 5, rightJaw.y);
    this.ctx.quadraticCurveTo(chin.x, chin.y + 10, leftJaw.x - 5, leftJaw.y);
    this.ctx.closePath();
  }

  // Draw phantom mask shape (half face)
  drawPhantomMaskShape(landmarks, canvasWidth, canvasHeight) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    const forehead = toPixel(10);
    const leftTemple = toPixel(127);
    const leftCheek = toPixel(234);
    const noseBridge = toPixel(6);
    const noseTip = toPixel(1);
    const leftEyebrowOuter = toPixel(46);
    const chin = toPixel(152);
    const leftJaw = toPixel(172);

    // Right half of face only
    this.ctx.moveTo(forehead.x, forehead.y - 15);
    this.ctx.quadraticCurveTo(leftEyebrowOuter.x, forehead.y - 10, leftTemple.x - 10, leftTemple.y);
    this.ctx.quadraticCurveTo(leftCheek.x - 15, leftCheek.y, leftJaw.x - 5, leftJaw.y);
    this.ctx.quadraticCurveTo(chin.x - 20, chin.y, noseTip.x, noseTip.y + 10);
    this.ctx.lineTo(noseBridge.x, noseBridge.y);
    this.ctx.lineTo(forehead.x, forehead.y - 15);
  }

  // Draw eye holes for mask
  drawEyeHoles(landmarks, canvasWidth, canvasHeight, shape, size) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    // Eye center positions
    const leftEyeCenter = {
      x: (toPixel(33).x + toPixel(133).x) / 2,
      y: (toPixel(159).y + toPixel(145).y) / 2
    };
    const rightEyeCenter = {
      x: (toPixel(263).x + toPixel(362).x) / 2,
      y: (toPixel(386).y + toPixel(374).y) / 2
    };

    // Eye dimensions
    const leftEyeWidth = Math.abs(toPixel(33).x - toPixel(133).x) * 0.7 * size;
    const leftEyeHeight = Math.abs(toPixel(159).y - toPixel(145).y) * 1.2 * size;
    const rightEyeWidth = Math.abs(toPixel(263).x - toPixel(362).x) * 0.7 * size;
    const rightEyeHeight = Math.abs(toPixel(386).y - toPixel(374).y) * 1.2 * size;

    // Calculate eye angles
    const leftAngle = Math.atan2(toPixel(133).y - toPixel(33).y, toPixel(133).x - toPixel(33).x);
    const rightAngle = Math.atan2(toPixel(362).y - toPixel(263).y, toPixel(362).x - toPixel(263).x);

    if (shape === 'pointed') {
      // Pointed/almond shaped holes
      this.drawPointedEyeHole(leftEyeCenter, leftEyeWidth, leftEyeHeight, leftAngle);
      this.drawPointedEyeHole(rightEyeCenter, rightEyeWidth, rightEyeHeight, rightAngle);
    } else if (shape === 'narrow') {
      // Narrow slits
      this.ctx.beginPath();
      this.ctx.ellipse(leftEyeCenter.x, leftEyeCenter.y, leftEyeWidth, leftEyeHeight * 0.5, leftAngle, 0, Math.PI * 2);
      this.ctx.fill();
      this.ctx.beginPath();
      this.ctx.ellipse(rightEyeCenter.x, rightEyeCenter.y, rightEyeWidth, rightEyeHeight * 0.5, rightAngle, 0, Math.PI * 2);
      this.ctx.fill();
    } else {
      // Oval or round
      const heightMult = shape === 'round' ? 1.0 : 0.7;
      this.ctx.beginPath();
      this.ctx.ellipse(leftEyeCenter.x, leftEyeCenter.y, leftEyeWidth, leftEyeHeight * heightMult, leftAngle, 0, Math.PI * 2);
      this.ctx.fill();
      this.ctx.beginPath();
      this.ctx.ellipse(rightEyeCenter.x, rightEyeCenter.y, rightEyeWidth, rightEyeHeight * heightMult, rightAngle, 0, Math.PI * 2);
      this.ctx.fill();
    }
  }

  // Draw pointed/almond eye hole
  drawPointedEyeHole(center, width, height, angle) {
    this.ctx.save();
    this.ctx.translate(center.x, center.y);
    this.ctx.rotate(angle);
    
    this.ctx.beginPath();
    this.ctx.moveTo(-width, 0);
    this.ctx.quadraticCurveTo(-width * 0.5, -height, 0, -height * 0.8);
    this.ctx.quadraticCurveTo(width * 0.5, -height, width, 0);
    this.ctx.quadraticCurveTo(width * 0.5, height, 0, height * 0.8);
    this.ctx.quadraticCurveTo(-width * 0.5, height, -width, 0);
    this.ctx.fill();
    
    this.ctx.restore();
  }

  // Draw eye hole borders
  drawEyeHoleBorders(landmarks, canvasWidth, canvasHeight, shape, size) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    const leftEyeCenter = {
      x: (toPixel(33).x + toPixel(133).x) / 2,
      y: (toPixel(159).y + toPixel(145).y) / 2
    };
    const rightEyeCenter = {
      x: (toPixel(263).x + toPixel(362).x) / 2,
      y: (toPixel(386).y + toPixel(374).y) / 2
    };

    const leftEyeWidth = Math.abs(toPixel(33).x - toPixel(133).x) * 0.7 * size;
    const leftEyeHeight = Math.abs(toPixel(159).y - toPixel(145).y) * 1.2 * size;
    const rightEyeWidth = Math.abs(toPixel(263).x - toPixel(362).x) * 0.7 * size;
    const rightEyeHeight = Math.abs(toPixel(386).y - toPixel(374).y) * 1.2 * size;

    const leftAngle = Math.atan2(toPixel(133).y - toPixel(33).y, toPixel(133).x - toPixel(33).x);
    const rightAngle = Math.atan2(toPixel(362).y - toPixel(263).y, toPixel(362).x - toPixel(263).x);

    if (shape === 'pointed') {
      this.drawPointedEyeHoleBorder(leftEyeCenter, leftEyeWidth, leftEyeHeight, leftAngle);
      this.drawPointedEyeHoleBorder(rightEyeCenter, rightEyeWidth, rightEyeHeight, rightAngle);
    } else if (shape === 'narrow') {
      this.ctx.beginPath();
      this.ctx.ellipse(leftEyeCenter.x, leftEyeCenter.y, leftEyeWidth, leftEyeHeight * 0.5, leftAngle, 0, Math.PI * 2);
      this.ctx.stroke();
      this.ctx.beginPath();
      this.ctx.ellipse(rightEyeCenter.x, rightEyeCenter.y, rightEyeWidth, rightEyeHeight * 0.5, rightAngle, 0, Math.PI * 2);
      this.ctx.stroke();
    } else {
      const heightMult = shape === 'round' ? 1.0 : 0.7;
      this.ctx.beginPath();
      this.ctx.ellipse(leftEyeCenter.x, leftEyeCenter.y, leftEyeWidth, leftEyeHeight * heightMult, leftAngle, 0, Math.PI * 2);
      this.ctx.stroke();
      this.ctx.beginPath();
      this.ctx.ellipse(rightEyeCenter.x, rightEyeCenter.y, rightEyeWidth, rightEyeHeight * heightMult, rightAngle, 0, Math.PI * 2);
      this.ctx.stroke();
    }
  }

  // Draw pointed eye hole border
  drawPointedEyeHoleBorder(center, width, height, angle) {
    this.ctx.save();
    this.ctx.translate(center.x, center.y);
    this.ctx.rotate(angle);
    
    this.ctx.beginPath();
    this.ctx.moveTo(-width, 0);
    this.ctx.quadraticCurveTo(-width * 0.5, -height, 0, -height * 0.8);
    this.ctx.quadraticCurveTo(width * 0.5, -height, width, 0);
    this.ctx.quadraticCurveTo(width * 0.5, height, 0, height * 0.8);
    this.ctx.quadraticCurveTo(-width * 0.5, height, -width, 0);
    this.ctx.stroke();
    
    this.ctx.restore();
  }

  // =====================================================
  // ACCESSORY OVERLAY SYSTEM
  // =====================================================

  // Draw accessory overlay (sunglasses, hats, etc.)
  drawAccessory(landmarks, canvasWidth, canvasHeight, settings = {}) {
    if (!landmarks || landmarks.length < 468) return;

    const {
      type = 'none',
      color = '#000000',
      opacity = 1.0,
      scale = 1.0
    } = settings;

    if (type === 'none') return;

    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    this.ctx.save();
    this.ctx.globalAlpha = opacity;

    switch (type) {
      case 'sunglasses':
        this.drawSunglasses(landmarks, canvasWidth, canvasHeight, color, scale);
        break;
      case 'aviators':
        this.drawAviators(landmarks, canvasWidth, canvasHeight, color, scale);
        break;
      case 'hearts':
        this.drawHeartGlasses(landmarks, canvasWidth, canvasHeight, color, scale);
        break;
      case 'stars':
        this.drawStarGlasses(landmarks, canvasWidth, canvasHeight, color, scale);
        break;
      case 'crown':
        this.drawCrown(landmarks, canvasWidth, canvasHeight, color, scale);
        break;
      case 'cat_ears':
        this.drawCatEars(landmarks, canvasWidth, canvasHeight, color, scale);
        break;
      case 'dog_ears':
        this.drawDogEars(landmarks, canvasWidth, canvasHeight, color, scale);
        break;
      case 'hat':
        this.drawTopHat(landmarks, canvasWidth, canvasHeight, color, scale);
        break;
    }

    this.ctx.restore();
  }

  // Draw classic sunglasses
  drawSunglasses(landmarks, canvasWidth, canvasHeight, color, scale) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    const leftEye = toPixel(33);
    const rightEye = toPixel(263);
    const noseBridge = toPixel(6);
    const leftTemple = toPixel(127);
    const rightTemple = toPixel(356);

    const eyeDistance = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y);
    const lensSize = eyeDistance * 0.45 * scale;
    const angle = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

    this.ctx.save();
    this.ctx.translate((leftEye.x + rightEye.x) / 2, (leftEye.y + rightEye.y) / 2);
    this.ctx.rotate(angle);

    // Frame
    this.ctx.strokeStyle = '#1a1a1a';
    this.ctx.lineWidth = 4 * scale;
    this.ctx.fillStyle = color;

    // Left lens
    this.ctx.beginPath();
    this.ctx.ellipse(-eyeDistance * 0.25, 0, lensSize, lensSize * 0.7, 0, 0, Math.PI * 2);
    this.ctx.fill();
    this.ctx.stroke();

    // Right lens
    this.ctx.beginPath();
    this.ctx.ellipse(eyeDistance * 0.25, 0, lensSize, lensSize * 0.7, 0, 0, Math.PI * 2);
    this.ctx.fill();
    this.ctx.stroke();

    // Bridge
    this.ctx.beginPath();
    this.ctx.moveTo(-eyeDistance * 0.25 + lensSize, 0);
    this.ctx.lineTo(eyeDistance * 0.25 - lensSize, 0);
    this.ctx.stroke();

    // Temple arms
    this.ctx.beginPath();
    this.ctx.moveTo(-eyeDistance * 0.25 - lensSize, 0);
    this.ctx.lineTo(-eyeDistance * 0.6, -lensSize * 0.3);
    this.ctx.moveTo(eyeDistance * 0.25 + lensSize, 0);
    this.ctx.lineTo(eyeDistance * 0.6, -lensSize * 0.3);
    this.ctx.stroke();

    // Lens reflection
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.arc(-eyeDistance * 0.25 - lensSize * 0.3, -lensSize * 0.2, lensSize * 0.2, 0, Math.PI * 0.8);
    this.ctx.stroke();
    this.ctx.beginPath();
    this.ctx.arc(eyeDistance * 0.25 - lensSize * 0.3, -lensSize * 0.2, lensSize * 0.2, 0, Math.PI * 0.8);
    this.ctx.stroke();

    this.ctx.restore();
  }

  // Draw aviator sunglasses
  drawAviators(landmarks, canvasWidth, canvasHeight, color, scale) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    const leftEye = toPixel(33);
    const rightEye = toPixel(263);
    const eyeDistance = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y);
    const angle = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

    this.ctx.save();
    this.ctx.translate((leftEye.x + rightEye.x) / 2, (leftEye.y + rightEye.y) / 2);
    this.ctx.rotate(angle);

    const lensW = eyeDistance * 0.4 * scale;
    const lensH = eyeDistance * 0.35 * scale;

    // Gold frame
    this.ctx.strokeStyle = '#c9a227';
    this.ctx.lineWidth = 3 * scale;

    // Gradient lens
    const gradient = this.ctx.createLinearGradient(0, -lensH, 0, lensH);
    gradient.addColorStop(0, 'rgba(50, 50, 50, 0.9)');
    gradient.addColorStop(0.5, color);
    gradient.addColorStop(1, 'rgba(80, 80, 80, 0.8)');
    this.ctx.fillStyle = gradient;

    // Teardrop shape - left
    this.ctx.beginPath();
    this.ctx.moveTo(-eyeDistance * 0.25, -lensH * 0.8);
    this.ctx.bezierCurveTo(-eyeDistance * 0.25 - lensW, -lensH, -eyeDistance * 0.25 - lensW, lensH * 0.5, -eyeDistance * 0.25, lensH);
    this.ctx.bezierCurveTo(-eyeDistance * 0.25 + lensW * 0.8, lensH * 0.5, -eyeDistance * 0.25 + lensW * 0.8, -lensH, -eyeDistance * 0.25, -lensH * 0.8);
    this.ctx.fill();
    this.ctx.stroke();

    // Teardrop shape - right
    this.ctx.beginPath();
    this.ctx.moveTo(eyeDistance * 0.25, -lensH * 0.8);
    this.ctx.bezierCurveTo(eyeDistance * 0.25 + lensW, -lensH, eyeDistance * 0.25 + lensW, lensH * 0.5, eyeDistance * 0.25, lensH);
    this.ctx.bezierCurveTo(eyeDistance * 0.25 - lensW * 0.8, lensH * 0.5, eyeDistance * 0.25 - lensW * 0.8, -lensH, eyeDistance * 0.25, -lensH * 0.8);
    this.ctx.fill();
    this.ctx.stroke();

    // Double bridge
    this.ctx.beginPath();
    this.ctx.moveTo(-eyeDistance * 0.1, -lensH * 0.3);
    this.ctx.lineTo(eyeDistance * 0.1, -lensH * 0.3);
    this.ctx.moveTo(-eyeDistance * 0.08, -lensH * 0.1);
    this.ctx.lineTo(eyeDistance * 0.08, -lensH * 0.1);
    this.ctx.stroke();

    this.ctx.restore();
  }

  // Draw heart-shaped glasses
  drawHeartGlasses(landmarks, canvasWidth, canvasHeight, color, scale) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    const leftEye = toPixel(33);
    const rightEye = toPixel(263);
    const eyeDistance = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y);
    const angle = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);
    const heartSize = eyeDistance * 0.35 * scale;

    this.ctx.save();
    this.ctx.translate((leftEye.x + rightEye.x) / 2, (leftEye.y + rightEye.y) / 2);
    this.ctx.rotate(angle);

    this.ctx.fillStyle = color || '#ff1493';
    this.ctx.strokeStyle = '#ff69b4';
    this.ctx.lineWidth = 2 * scale;

    // Draw heart at each eye position
    [-eyeDistance * 0.25, eyeDistance * 0.25].forEach(xOffset => {
      this.ctx.beginPath();
      this.ctx.moveTo(xOffset, heartSize * 0.3);
      this.ctx.bezierCurveTo(xOffset - heartSize, -heartSize * 0.3, xOffset - heartSize * 0.5, -heartSize, xOffset, -heartSize * 0.5);
      this.ctx.bezierCurveTo(xOffset + heartSize * 0.5, -heartSize, xOffset + heartSize, -heartSize * 0.3, xOffset, heartSize * 0.3);
      this.ctx.fill();
      this.ctx.stroke();
    });

    // Bridge
    this.ctx.strokeStyle = '#ff69b4';
    this.ctx.lineWidth = 3 * scale;
    this.ctx.beginPath();
    this.ctx.moveTo(-eyeDistance * 0.25 + heartSize * 0.8, -heartSize * 0.2);
    this.ctx.lineTo(eyeDistance * 0.25 - heartSize * 0.8, -heartSize * 0.2);
    this.ctx.stroke();

    this.ctx.restore();
  }

  // Draw star-shaped glasses
  drawStarGlasses(landmarks, canvasWidth, canvasHeight, color, scale) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    const leftEye = toPixel(33);
    const rightEye = toPixel(263);
    const eyeDistance = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y);
    const angle = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);
    const starSize = eyeDistance * 0.35 * scale;

    this.ctx.save();
    this.ctx.translate((leftEye.x + rightEye.x) / 2, (leftEye.y + rightEye.y) / 2);
    this.ctx.rotate(angle);

    this.ctx.fillStyle = color || '#ffd700';
    this.ctx.strokeStyle = '#ff8c00';
    this.ctx.lineWidth = 2 * scale;

    // Draw star at each eye
    [-eyeDistance * 0.25, eyeDistance * 0.25].forEach(xOffset => {
      this.drawStar(xOffset, 0, 5, starSize, starSize * 0.5);
    });

    // Bridge
    this.ctx.strokeStyle = '#ff8c00';
    this.ctx.lineWidth = 3 * scale;
    this.ctx.beginPath();
    this.ctx.moveTo(-eyeDistance * 0.25 + starSize, 0);
    this.ctx.lineTo(eyeDistance * 0.25 - starSize, 0);
    this.ctx.stroke();

    this.ctx.restore();
  }

  // Helper to draw a star shape
  drawStar(cx, cy, spikes, outerRadius, innerRadius) {
    let rot = Math.PI / 2 * 3;
    const step = Math.PI / spikes;

    this.ctx.beginPath();
    this.ctx.moveTo(cx, cy - outerRadius);

    for (let i = 0; i < spikes; i++) {
      let x = cx + Math.cos(rot) * outerRadius;
      let y = cy + Math.sin(rot) * outerRadius;
      this.ctx.lineTo(x, y);
      rot += step;

      x = cx + Math.cos(rot) * innerRadius;
      y = cy + Math.sin(rot) * innerRadius;
      this.ctx.lineTo(x, y);
      rot += step;
    }

    this.ctx.lineTo(cx, cy - outerRadius);
    this.ctx.closePath();
    this.ctx.fill();
    this.ctx.stroke();
  }

  // Draw crown
  drawCrown(landmarks, canvasWidth, canvasHeight, color, scale) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    const forehead = toPixel(10);
    const leftTemple = toPixel(127);
    const rightTemple = toPixel(356);
    const leftEye = toPixel(33);
    const rightEye = toPixel(263);

    const crownWidth = Math.abs(rightTemple.x - leftTemple.x) * 1.1 * scale;
    const crownHeight = crownWidth * 0.5;
    const angle = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

    this.ctx.save();
    this.ctx.translate(forehead.x, forehead.y - crownHeight * 0.6);
    this.ctx.rotate(angle);

    // Crown gradient
    const gradient = this.ctx.createLinearGradient(0, -crownHeight, 0, crownHeight * 0.3);
    gradient.addColorStop(0, '#ffd700');
    gradient.addColorStop(0.5, color || '#ffb347');
    gradient.addColorStop(1, '#ff8c00');
    this.ctx.fillStyle = gradient;

    // Crown shape
    this.ctx.beginPath();
    this.ctx.moveTo(-crownWidth / 2, crownHeight * 0.3);
    this.ctx.lineTo(-crownWidth / 2, 0);
    this.ctx.lineTo(-crownWidth * 0.35, -crownHeight * 0.5);
    this.ctx.lineTo(-crownWidth * 0.2, 0);
    this.ctx.lineTo(0, -crownHeight);
    this.ctx.lineTo(crownWidth * 0.2, 0);
    this.ctx.lineTo(crownWidth * 0.35, -crownHeight * 0.5);
    this.ctx.lineTo(crownWidth / 2, 0);
    this.ctx.lineTo(crownWidth / 2, crownHeight * 0.3);
    this.ctx.closePath();
    this.ctx.fill();

    // Crown outline
    this.ctx.strokeStyle = '#b8860b';
    this.ctx.lineWidth = 2 * scale;
    this.ctx.stroke();

    // Jewels
    this.ctx.fillStyle = '#ff0000';
    this.ctx.beginPath();
    this.ctx.arc(0, -crownHeight * 0.65, crownHeight * 0.12, 0, Math.PI * 2);
    this.ctx.fill();

    this.ctx.fillStyle = '#0000ff';
    this.ctx.beginPath();
    this.ctx.arc(-crownWidth * 0.35, -crownHeight * 0.25, crownHeight * 0.08, 0, Math.PI * 2);
    this.ctx.fill();
    this.ctx.beginPath();
    this.ctx.arc(crownWidth * 0.35, -crownHeight * 0.25, crownHeight * 0.08, 0, Math.PI * 2);
    this.ctx.fill();

    this.ctx.restore();
  }

  // Draw cat ears
  drawCatEars(landmarks, canvasWidth, canvasHeight, color, scale) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    const forehead = toPixel(10);
    const leftTemple = toPixel(127);
    const rightTemple = toPixel(356);
    const leftEye = toPixel(33);
    const rightEye = toPixel(263);

    const earSize = Math.abs(rightTemple.x - leftTemple.x) * 0.3 * scale;
    const angle = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

    this.ctx.save();

    // Left ear
    this.ctx.translate(leftTemple.x, forehead.y - earSize * 0.5);
    this.ctx.rotate(angle - 0.3);
    this.drawCatEar(earSize, color);
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);

    // Right ear
    this.ctx.translate(rightTemple.x, forehead.y - earSize * 0.5);
    this.ctx.rotate(angle + 0.3);
    this.drawCatEar(earSize, color);

    this.ctx.restore();
  }

  drawCatEar(size, color) {
    // Outer ear
    this.ctx.beginPath();
    this.ctx.moveTo(0, size);
    this.ctx.lineTo(-size * 0.6, -size);
    this.ctx.lineTo(size * 0.6, -size);
    this.ctx.closePath();
    this.ctx.fillStyle = color || '#8b4513';
    this.ctx.fill();
    this.ctx.strokeStyle = '#5d3a1a';
    this.ctx.lineWidth = 2;
    this.ctx.stroke();

    // Inner ear
    this.ctx.beginPath();
    this.ctx.moveTo(0, size * 0.5);
    this.ctx.lineTo(-size * 0.35, -size * 0.6);
    this.ctx.lineTo(size * 0.35, -size * 0.6);
    this.ctx.closePath();
    this.ctx.fillStyle = '#ffb6c1';
    this.ctx.fill();
  }

  // Draw dog/puppy ears
  drawDogEars(landmarks, canvasWidth, canvasHeight, color, scale) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    const forehead = toPixel(10);
    const leftTemple = toPixel(127);
    const rightTemple = toPixel(356);
    const leftCheek = toPixel(234);
    const rightCheek = toPixel(454);
    const leftEye = toPixel(33);
    const rightEye = toPixel(263);

    const earWidth = Math.abs(rightTemple.x - leftTemple.x) * 0.35 * scale;
    const earHeight = earWidth * 1.5;
    const angle = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

    this.ctx.save();

    const baseColor = color || '#8b4513';

    // Left floppy ear
    this.ctx.translate(leftTemple.x - earWidth * 0.3, forehead.y);
    this.ctx.rotate(angle - 0.5);
    this.ctx.beginPath();
    this.ctx.ellipse(0, earHeight * 0.5, earWidth * 0.5, earHeight, 0, 0, Math.PI * 2);
    this.ctx.fillStyle = baseColor;
    this.ctx.fill();
    this.ctx.strokeStyle = '#5d3a1a';
    this.ctx.lineWidth = 2;
    this.ctx.stroke();
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);

    // Right floppy ear
    this.ctx.translate(rightTemple.x + earWidth * 0.3, forehead.y);
    this.ctx.rotate(angle + 0.5);
    this.ctx.beginPath();
    this.ctx.ellipse(0, earHeight * 0.5, earWidth * 0.5, earHeight, 0, 0, Math.PI * 2);
    this.ctx.fillStyle = baseColor;
    this.ctx.fill();
    this.ctx.stroke();

    this.ctx.restore();
  }

  // Draw top hat
  drawTopHat(landmarks, canvasWidth, canvasHeight, color, scale) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    const forehead = toPixel(10);
    const leftTemple = toPixel(127);
    const rightTemple = toPixel(356);
    const leftEye = toPixel(33);
    const rightEye = toPixel(263);

    const hatWidth = Math.abs(rightTemple.x - leftTemple.x) * 0.9 * scale;
    const hatHeight = hatWidth * 0.8;
    const angle = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

    this.ctx.save();
    this.ctx.translate(forehead.x, forehead.y - hatHeight * 0.4);
    this.ctx.rotate(angle);

    const baseColor = color || '#1a1a1a';

    // Brim
    this.ctx.beginPath();
    this.ctx.ellipse(0, hatHeight * 0.3, hatWidth * 0.7, hatHeight * 0.15, 0, 0, Math.PI * 2);
    this.ctx.fillStyle = baseColor;
    this.ctx.fill();
    this.ctx.strokeStyle = '#333';
    this.ctx.lineWidth = 2;
    this.ctx.stroke();

    // Top cylinder
    this.ctx.beginPath();
    this.ctx.rect(-hatWidth * 0.35, -hatHeight * 0.6, hatWidth * 0.7, hatHeight * 0.9);
    this.ctx.fillStyle = baseColor;
    this.ctx.fill();
    this.ctx.stroke();

    // Top
    this.ctx.beginPath();
    this.ctx.ellipse(0, -hatHeight * 0.6, hatWidth * 0.35, hatHeight * 0.1, 0, 0, Math.PI * 2);
    this.ctx.fill();
    this.ctx.stroke();

    // Band
    this.ctx.fillStyle = '#8b0000';
    this.ctx.fillRect(-hatWidth * 0.35, hatHeight * 0.1, hatWidth * 0.7, hatHeight * 0.1);

    this.ctx.restore();
  }

  // =====================================================
  // MAKEUP FILTERS
  // =====================================================

  // Draw makeup effects
  drawMakeup(landmarks, canvasWidth, canvasHeight, settings = {}) {
    if (!landmarks || landmarks.length < 468) return;

    const {
      lipstickEnabled = false,
      lipstickColor = '#cc2244',
      lipstickOpacity = 0.7,
      lipstickGloss = true,
      eyelinerEnabled = false,
      eyelinerColor = '#000000',
      eyelinerWidth = 2,
      eyelinerStyle = 'classic',
      blushEnabled = false,
      blushColor = '#ff9999',
      blushOpacity = 0.3,
      eyeshadowEnabled = false,
      eyeshadowColor = '#8844aa',
      eyeshadowOpacity = 0.4
    } = settings;

    if (lipstickEnabled) {
      this.drawLipstick(landmarks, canvasWidth, canvasHeight, lipstickColor, lipstickOpacity, lipstickGloss);
    }

    if (eyelinerEnabled) {
      this.drawEyeliner(landmarks, canvasWidth, canvasHeight, eyelinerColor, eyelinerWidth, eyelinerStyle);
    }

    if (blushEnabled) {
      this.drawBlush(landmarks, canvasWidth, canvasHeight, blushColor, blushOpacity);
    }

    if (eyeshadowEnabled) {
      this.drawEyeshadow(landmarks, canvasWidth, canvasHeight, eyeshadowColor, eyeshadowOpacity);
    }
  }

  // Draw lipstick
  drawLipstick(landmarks, canvasWidth, canvasHeight, color, opacity, gloss) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    // Outer lip landmarks
    const outerLip = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
    // Inner lip landmarks
    const innerLip = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191];

    this.ctx.save();
    this.ctx.globalAlpha = opacity;

    // Fill outer lips
    this.ctx.beginPath();
    const first = toPixel(outerLip[0]);
    this.ctx.moveTo(first.x, first.y);
    for (let i = 1; i < outerLip.length; i++) {
      const p = toPixel(outerLip[i]);
      this.ctx.lineTo(p.x, p.y);
    }
    this.ctx.closePath();
    this.ctx.fillStyle = color;
    this.ctx.fill();

    // Add gloss highlight
    if (gloss) {
      const lipCenter = toPixel(0);
      const lipTop = toPixel(13);
      const glossGradient = this.ctx.createLinearGradient(lipCenter.x, lipTop.y, lipCenter.x, lipCenter.y);
      glossGradient.addColorStop(0, 'rgba(255, 255, 255, 0.4)');
      glossGradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.1)');
      glossGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
      
      this.ctx.globalCompositeOperation = 'overlay';
      this.ctx.fillStyle = glossGradient;
      this.ctx.fill();
      this.ctx.globalCompositeOperation = 'source-over';
    }

    this.ctx.restore();
  }

  // Draw eyeliner
  drawEyeliner(landmarks, canvasWidth, canvasHeight, color, width, style) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    this.ctx.save();
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = width;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';

    // Left eye upper lid
    const leftUpper = [33, 246, 161, 160, 159, 158, 157, 173, 133];
    // Right eye upper lid
    const rightUpper = [263, 466, 388, 387, 386, 385, 384, 398, 362];

    // Draw left eyeliner
    this.ctx.beginPath();
    leftUpper.forEach((idx, i) => {
      const p = toPixel(idx);
      if (i === 0) this.ctx.moveTo(p.x, p.y);
      else this.ctx.lineTo(p.x, p.y);
    });
    
    // Wing for cat-eye style
    if (style === 'wing' || style === 'smoky') {
      const lastPoint = toPixel(133);
      const wingEnd = { x: lastPoint.x + width * 4, y: lastPoint.y - width * 3 };
      this.ctx.lineTo(wingEnd.x, wingEnd.y);
    }
    this.ctx.stroke();

    // Draw right eyeliner
    this.ctx.beginPath();
    rightUpper.forEach((idx, i) => {
      const p = toPixel(idx);
      if (i === 0) this.ctx.moveTo(p.x, p.y);
      else this.ctx.lineTo(p.x, p.y);
    });
    
    if (style === 'wing' || style === 'smoky') {
      const lastPoint = toPixel(362);
      const wingEnd = { x: lastPoint.x - width * 4, y: lastPoint.y - width * 3 };
      this.ctx.lineTo(wingEnd.x, wingEnd.y);
    }
    this.ctx.stroke();

    // Smoky effect
    if (style === 'smoky') {
      this.ctx.globalAlpha = 0.3;
      this.ctx.lineWidth = width * 3;
      this.ctx.filter = 'blur(3px)';
      
      this.ctx.beginPath();
      leftUpper.forEach((idx, i) => {
        const p = toPixel(idx);
        if (i === 0) this.ctx.moveTo(p.x, p.y - 2);
        else this.ctx.lineTo(p.x, p.y - 2);
      });
      this.ctx.stroke();

      this.ctx.beginPath();
      rightUpper.forEach((idx, i) => {
        const p = toPixel(idx);
        if (i === 0) this.ctx.moveTo(p.x, p.y - 2);
        else this.ctx.lineTo(p.x, p.y - 2);
      });
      this.ctx.stroke();
      this.ctx.filter = 'none';
    }

    this.ctx.restore();
  }

  // Draw blush
  drawBlush(landmarks, canvasWidth, canvasHeight, color, opacity) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    // Cheek positions
    const leftCheek = toPixel(50);
    const rightCheek = toPixel(280);
    const leftEye = toPixel(33);
    const rightEye = toPixel(263);
    
    const blushSize = Math.abs(rightEye.x - leftEye.x) * 0.25;

    this.ctx.save();
    this.ctx.globalAlpha = opacity;

    // Create radial gradient for soft blush
    const drawBlushCircle = (center) => {
      const gradient = this.ctx.createRadialGradient(center.x, center.y, 0, center.x, center.y, blushSize);
      gradient.addColorStop(0, color);
      gradient.addColorStop(0.5, color);
      gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
      
      this.ctx.beginPath();
      this.ctx.arc(center.x, center.y, blushSize, 0, Math.PI * 2);
      this.ctx.fillStyle = gradient;
      this.ctx.fill();
    };

    drawBlushCircle(leftCheek);
    drawBlushCircle(rightCheek);

    this.ctx.restore();
  }

  // Draw eyeshadow
  drawEyeshadow(landmarks, canvasWidth, canvasHeight, color, opacity) {
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight
    });

    this.ctx.save();
    this.ctx.globalAlpha = opacity;

    // Left eye upper area
    const leftEyeArea = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 156, 143, 111, 117, 118, 119, 120, 121, 128, 245];
    // Right eye upper area  
    const rightEyeArea = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276, 383, 372, 340, 346, 347, 348, 349, 350, 357, 465];

    const drawEyeArea = (indices) => {
      if (indices.length < 3) return;
      this.ctx.beginPath();
      const first = toPixel(indices[0]);
      this.ctx.moveTo(first.x, first.y);
      for (let i = 1; i < indices.length; i++) {
        if (indices[i] < landmarks.length) {
          const p = toPixel(indices[i]);
          this.ctx.lineTo(p.x, p.y);
        }
      }
      this.ctx.closePath();
      this.ctx.fillStyle = color;
      this.ctx.fill();
    };

    drawEyeArea(leftEyeArea);
    drawEyeArea(rightEyeArea);

    this.ctx.restore();
  }

  // =====================================================
  // FACE MORPHING EFFECTS
  // =====================================================

  // Apply face morphing and draw result
  drawMorphedFace(landmarks, canvasWidth, canvasHeight, settings = {}) {
    if (!landmarks || landmarks.length < 468) return null;

    const {
      eyeSize = 1.0,
      noseSize = 1.0,
      faceWidth = 1.0,
      foreheadSize = 1.0,
      chinSize = 1.0
    } = settings;

    // If no morphing needed, return original landmarks
    if (eyeSize === 1.0 && noseSize === 1.0 && faceWidth === 1.0 && foreheadSize === 1.0 && chinSize === 1.0) {
      return landmarks;
    }

    // Create morphed copy of landmarks
    const morphed = landmarks.map(lm => ({ ...lm }));

    // Face center for reference
    const centerX = (landmarks[33].x + landmarks[263].x) / 2;
    const centerY = (landmarks[33].y + landmarks[263].y) / 2;

    // Eye landmarks
    const leftEyeLandmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
    const rightEyeLandmarks = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];
    
    // Scale eyes
    if (eyeSize !== 1.0) {
      const leftEyeCenter = { x: landmarks[159].x, y: landmarks[159].y };
      const rightEyeCenter = { x: landmarks[386].x, y: landmarks[386].y };

      leftEyeLandmarks.forEach(idx => {
        morphed[idx].x = leftEyeCenter.x + (landmarks[idx].x - leftEyeCenter.x) * eyeSize;
        morphed[idx].y = leftEyeCenter.y + (landmarks[idx].y - leftEyeCenter.y) * eyeSize;
      });

      rightEyeLandmarks.forEach(idx => {
        morphed[idx].x = rightEyeCenter.x + (landmarks[idx].x - rightEyeCenter.x) * eyeSize;
        morphed[idx].y = rightEyeCenter.y + (landmarks[idx].y - rightEyeCenter.y) * eyeSize;
      });
    }

    // Nose landmarks
    const noseLandmarks = [1, 2, 4, 5, 6, 19, 94, 97, 98, 129, 168, 195, 197, 326, 327, 358];
    
    if (noseSize !== 1.0) {
      const noseCenter = { x: landmarks[1].x, y: landmarks[1].y };
      
      noseLandmarks.forEach(idx => {
        morphed[idx].x = noseCenter.x + (landmarks[idx].x - noseCenter.x) * noseSize;
        morphed[idx].y = noseCenter.y + (landmarks[idx].y - noseCenter.y) * noseSize;
      });
    }

    // Face width - affect outer face landmarks
    if (faceWidth !== 1.0) {
      for (let i = 0; i < morphed.length; i++) {
        const distFromCenter = morphed[i].x - centerX;
        morphed[i].x = centerX + distFromCenter * faceWidth;
      }
    }

    // Forehead - move upper landmarks
    if (foreheadSize !== 1.0) {
      const foreheadLandmarks = [10, 67, 69, 103, 104, 108, 109, 151, 297, 299, 332, 333, 337, 338];
      const foreheadBase = landmarks[10].y;
      
      foreheadLandmarks.forEach(idx => {
        const distFromBase = landmarks[idx].y - foreheadBase;
        morphed[idx].y = foreheadBase + distFromBase * (2 - foreheadSize);
      });
    }

    // Chin - move lower landmarks
    if (chinSize !== 1.0) {
      const chinLandmarks = [152, 148, 176, 149, 150, 136, 172, 58, 132, 377, 378, 365, 397, 288, 361];
      const chinBase = landmarks[152].y;
      const mouthY = landmarks[13].y;
      
      chinLandmarks.forEach(idx => {
        const distFromMouth = landmarks[idx].y - mouthY;
        if (distFromMouth > 0) {
          morphed[idx].y = mouthY + distFromMouth * chinSize;
        }
      });
    }

    return morphed;
  }

  // Visualize morphing effect with colored overlay
  drawMorphVisualization(originalLandmarks, morphedLandmarks, canvasWidth, canvasHeight) {
    if (!originalLandmarks || !morphedLandmarks) return;

    this.ctx.save();
    this.ctx.globalAlpha = 0.3;
    this.ctx.strokeStyle = '#00ff00';
    this.ctx.lineWidth = 1;

    // Draw lines showing displacement
    for (let i = 0; i < Math.min(originalLandmarks.length, morphedLandmarks.length); i++) {
      const orig = {
        x: originalLandmarks[i].x * canvasWidth,
        y: originalLandmarks[i].y * canvasHeight
      };
      const morph = {
        x: morphedLandmarks[i].x * canvasWidth,
        y: morphedLandmarks[i].y * canvasHeight
      };

      if (Math.hypot(morph.x - orig.x, morph.y - orig.y) > 1) {
        this.ctx.beginPath();
        this.ctx.moveTo(orig.x, orig.y);
        this.ctx.lineTo(morph.x, morph.y);
        this.ctx.stroke();
      }
    }

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
      showFaceOval = true,
      // Adaptive LOD options
      useAdaptiveLOD = false,
      enableDenseLandmarks = true,
      enableSubdivision = true,
      lodLevel = 1.0,
      enableSymmetry = true,
      showRegionColors = false  // Color triangles by facial region
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

    // Collect all triangles - use adaptive LOD or MediaPipe's triangulation
    const triangles = [];
    let triangleIndices;
    let processedLandmarks = landmarks;
    
    if (useAdaptiveLOD) {
      // Use adaptive Delaunay triangulation with LOD
      // Configure dense landmarks and subdivision (inspired by Wood et al. 2022)
      adaptiveTriangulation.setLOD(lodLevel);
      adaptiveTriangulation.enableDenseLandmarks = enableDenseLandmarks;
      adaptiveTriangulation.enableSubdivision = enableSubdivision;
      adaptiveTriangulation.updateSettings({ enableSymmetry });
      
      const result = adaptiveTriangulation.generateAdaptiveTriangulation(landmarks);
      triangleIndices = result.triangles; // Array of [i0, i1, i2] arrays
      processedLandmarks = result.landmarks;
      
      for (const [idx0, idx1, idx2] of triangleIndices) {
        if (idx0 >= processedLandmarks.length || idx1 >= processedLandmarks.length || idx2 >= processedLandmarks.length) {
          continue;
        }

        const p0 = {
          x: processedLandmarks[idx0].x * canvasWidth,
          y: processedLandmarks[idx0].y * canvasHeight,
          z: processedLandmarks[idx0].z || 0
        };
        const p1 = {
          x: processedLandmarks[idx1].x * canvasWidth,
          y: processedLandmarks[idx1].y * canvasHeight,
          z: processedLandmarks[idx1].z || 0
        };
        const p2 = {
          x: processedLandmarks[idx2].x * canvasWidth,
          y: processedLandmarks[idx2].y * canvasHeight,
          z: processedLandmarks[idx2].z || 0
        };
        
        const avgZ = (p0.z + p1.z + p2.z) / 3;
        
        // Get region info for coloring
        const regionInfo = showRegionColors ? adaptiveTriangulation.getRegionInfo(idx0) : null;
        
        triangles.push({ p0, p1, p2, avgZ, idx0, idx1, idx2, regionInfo });
      }
    } else {
      // Use MediaPipe's predefined triangulation
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
        
        const avgZ = (p0.z + p1.z + p2.z) / 3;
        
        triangles.push({ p0, p1, p2, avgZ, idx0, idx1, idx2, regionInfo: null });
      }
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
    
    // Region colors for visualization (when showRegionColors is enabled)
    const regionColors = {
      LEFT_EYE: { r: 255, g: 100, b: 100 },      // Red
      RIGHT_EYE: { r: 255, g: 100, b: 100 },     // Red
      LEFT_EYEBROW: { r: 255, g: 180, b: 0 },    // Orange
      RIGHT_EYEBROW: { r: 255, g: 180, b: 0 },   // Orange
      UPPER_LIP: { r: 255, g: 50, b: 150 },      // Pink
      LOWER_LIP: { r: 255, g: 50, b: 150 },      // Pink
      NOSE: { r: 100, g: 255, b: 100 },          // Green
      LEFT_CHEEK: { r: 100, g: 200, b: 255 },    // Light Blue
      RIGHT_CHEEK: { r: 100, g: 200, b: 255 },   // Light Blue
      CHIN: { r: 200, g: 100, b: 255 },          // Purple
      FOREHEAD: { r: 255, g: 255, b: 100 },      // Yellow
      FACE_OVAL: { r: 150, g: 150, b: 255 },     // Light Purple
      OTHER: { r: 200, g: 200, b: 200 }          // Gray
    };

    if (showTriangles) {
      // Calculate median triangle area for more robust filtering
      // (Skip this for adaptive LOD since it already filters)
      let minTriangleArea = 0;
      let maxTriangleArea = Infinity;
      
      if (!useAdaptiveLOD) {
        const areas = triangles.map(tri => {
          const { p0, p1, p2 } = tri;
          return Math.abs((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)) / 2;
        });
        areas.sort((a, b) => a - b);
        const medianArea = areas[Math.floor(areas.length / 2)];
        
        // More aggressive filtering - use median-based thresholds
        minTriangleArea = medianArea * 0.3;   // Skip if < 30% of median
        maxTriangleArea = medianArea * 2.5;   // Skip if > 250% of median
      }
      
      for (const tri of triangles) {
        const { p0, p1, p2, avgZ, regionInfo } = tri;
        
        // Calculate triangle area and skip outliers (only for non-adaptive mode)
        if (!useAdaptiveLOD) {
          const area = Math.abs((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)) / 2;
          if (area < minTriangleArea || area > maxTriangleArea) continue;
        }
        
        this.ctx.beginPath();
        this.ctx.moveTo(p0.x, p0.y);
        this.ctx.lineTo(p1.x, p1.y);
        this.ctx.lineTo(p2.x, p2.y);
        this.ctx.closePath();

        if (showRegionColors && regionInfo) {
          // Color by facial region
          const color = regionColors[regionInfo.region] || regionColors.OTHER;
          this.ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${fillOpacity * 3})`;
          this.ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.6)`;
          if (fillOpacity > 0) this.ctx.fill();
        } else if (useDepth) {
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

  // Draw iris tracking visualization (landmarks 468-477)
  // 468-472: Left iris (center + 4 cardinal points)
  // 473-477: Right iris (center + 4 cardinal points)
  drawIrisTracking(landmarks, canvasWidth, canvasHeight, options = {}) {
    // Check if iris landmarks exist (478 total landmarks)
    if (!landmarks || landmarks.length < 478) {
      return;
    }

    const {
      irisColor = '#00ffff',
      pupilColor = '#ffffff',
      showIrisCircle = true,
      showPupil = true,
      showGazeDirection = true,
      irisOpacity = 0.8
    } = options;

    this.ctx.save();

    // Iris landmark indices
    const LEFT_IRIS = {
      center: 468,
      right: 469,  // +X direction
      top: 470,    // -Y direction  
      left: 471,   // -X direction
      bottom: 472  // +Y direction
    };

    const RIGHT_IRIS = {
      center: 473,
      right: 474,
      top: 475,
      left: 476,
      bottom: 477
    };

    // Helper to convert normalized coords to pixels
    const toPixel = (idx) => ({
      x: landmarks[idx].x * canvasWidth,
      y: landmarks[idx].y * canvasHeight,
      z: landmarks[idx].z || 0
    });

    // Draw iris for each eye
    for (const iris of [LEFT_IRIS, RIGHT_IRIS]) {
      const center = toPixel(iris.center);
      const right = toPixel(iris.right);
      const top = toPixel(iris.top);
      const left = toPixel(iris.left);
      const bottom = toPixel(iris.bottom);

      // Calculate iris radius (average of horizontal and vertical)
      const radiusH = (Math.hypot(right.x - center.x, right.y - center.y) +
                       Math.hypot(left.x - center.x, left.y - center.y)) / 2;
      const radiusV = (Math.hypot(top.x - center.x, top.y - center.y) +
                       Math.hypot(bottom.x - center.x, bottom.y - center.y)) / 2;
      const irisRadius = (radiusH + radiusV) / 2;

      if (showIrisCircle) {
        // Draw iris outer ring
        this.ctx.beginPath();
        this.ctx.arc(center.x, center.y, irisRadius, 0, Math.PI * 2);
        this.ctx.strokeStyle = irisColor;
        this.ctx.lineWidth = 2;
        this.ctx.globalAlpha = irisOpacity;
        this.ctx.stroke();

        // Draw iris fill (subtle)
        const gradient = this.ctx.createRadialGradient(
          center.x, center.y, 0,
          center.x, center.y, irisRadius
        );
        gradient.addColorStop(0, 'rgba(0, 200, 255, 0.3)');
        gradient.addColorStop(0.7, 'rgba(0, 150, 200, 0.2)');
        gradient.addColorStop(1, 'rgba(0, 100, 150, 0.1)');
        this.ctx.fillStyle = gradient;
        this.ctx.fill();
      }

      if (showPupil) {
        // Draw pupil (center point)
        const pupilRadius = irisRadius * 0.35;
        this.ctx.beginPath();
        this.ctx.arc(center.x, center.y, pupilRadius, 0, Math.PI * 2);
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        this.ctx.fill();

        // Pupil highlight
        this.ctx.beginPath();
        this.ctx.arc(center.x - pupilRadius * 0.3, center.y - pupilRadius * 0.3, 
                     pupilRadius * 0.25, 0, Math.PI * 2);
        this.ctx.fillStyle = pupilColor;
        this.ctx.globalAlpha = 0.6;
        this.ctx.fill();
      }

      // Draw the 4 cardinal iris points
      this.ctx.globalAlpha = irisOpacity;
      this.ctx.fillStyle = irisColor;
      for (const point of [right, top, left, bottom]) {
        this.ctx.beginPath();
        this.ctx.arc(point.x, point.y, 2, 0, Math.PI * 2);
        this.ctx.fill();
      }
    }

    // Draw gaze direction lines extending from eyes
    if (showGazeDirection) {
      const leftCenter = toPixel(LEFT_IRIS.center);
      const rightCenter = toPixel(RIGHT_IRIS.center);

      // Get eye corners for reference
      const leftInner = landmarks[133] ? {
        x: landmarks[133].x * canvasWidth,
        y: landmarks[133].y * canvasHeight
      } : leftCenter;
      const leftOuter = landmarks[33] ? {
        x: landmarks[33].x * canvasWidth,
        y: landmarks[33].y * canvasHeight
      } : leftCenter;
      
      const rightInner = landmarks[362] ? {
        x: landmarks[362].x * canvasWidth,
        y: landmarks[362].y * canvasHeight
      } : rightCenter;
      const rightOuter = landmarks[263] ? {
        x: landmarks[263].x * canvasWidth,
        y: landmarks[263].y * canvasHeight
      } : rightCenter;

      // Calculate eye centers (midpoint of inner/outer corners)
      const leftEyeCenter = {
        x: (leftInner.x + leftOuter.x) / 2,
        y: (leftInner.y + leftOuter.y) / 2
      };
      const rightEyeCenter = {
        x: (rightInner.x + rightOuter.x) / 2,
        y: (rightInner.y + rightOuter.y) / 2
      };

      // Calculate gaze offset (iris position relative to eye center)
      const leftOffset = {
        x: leftCenter.x - leftEyeCenter.x,
        y: leftCenter.y - leftEyeCenter.y
      };
      const rightOffset = {
        x: rightCenter.x - rightEyeCenter.x,
        y: rightCenter.y - rightEyeCenter.y
      };

      // Draw gaze lines extending from each iris
      this.ctx.globalAlpha = 0.8;
      this.ctx.strokeStyle = '#ff00ff';
      this.ctx.lineWidth = 2;
      this.ctx.lineCap = 'round';

      const gazeLength = 60; // Length of gaze line
      
      for (const [irisCenter, offset] of [[leftCenter, leftOffset], [rightCenter, rightOffset]]) {
        // Normalize and extend the gaze direction
        const magnitude = Math.sqrt(offset.x * offset.x + offset.y * offset.y);
        if (magnitude > 0.5) { // Only draw if there's noticeable gaze offset
          const dirX = offset.x / magnitude;
          const dirY = offset.y / magnitude;
          
          // Draw gaze ray
          this.ctx.beginPath();
          this.ctx.moveTo(irisCenter.x, irisCenter.y);
          this.ctx.lineTo(
            irisCenter.x + dirX * gazeLength * (magnitude * 3),
            irisCenter.y + dirY * gazeLength * (magnitude * 3)
          );
          this.ctx.stroke();
          
          // Draw arrowhead
          const arrowX = irisCenter.x + dirX * gazeLength * (magnitude * 3);
          const arrowY = irisCenter.y + dirY * gazeLength * (magnitude * 3);
          const arrowSize = 6;
          
          this.ctx.fillStyle = '#ff00ff';
          this.ctx.beginPath();
          this.ctx.moveTo(arrowX, arrowY);
          this.ctx.lineTo(arrowX - dirX * arrowSize - dirY * arrowSize * 0.5, 
                         arrowY - dirY * arrowSize + dirX * arrowSize * 0.5);
          this.ctx.lineTo(arrowX - dirX * arrowSize + dirY * arrowSize * 0.5, 
                         arrowY - dirY * arrowSize - dirX * arrowSize * 0.5);
          this.ctx.closePath();
          this.ctx.fill();
        }
      }
    }

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

    this.ctx.save();

    // Collect detected expressions
    const detectedExpressions = [];

    if (expressions.smiling) {
      detectedExpressions.push({ text: 'Smiling', color: '#FFD700', intensity: expressions.smileAmount });
    }
    if (expressions.mouthOpen) {
      detectedExpressions.push({ text: 'Mouth Open', color: '#FF6B6B', intensity: expressions.mouthOpenAmount });
    }
    if (expressions.surprised) {
      detectedExpressions.push({ text: 'Surprised', color: '#00BFFF', intensity: expressions.browRaiseAmount });
    }
    if (expressions.bothEyesClosed) {
      detectedExpressions.push({ text: 'Eyes Closed', color: '#9370DB', intensity: 1 });
    } else if (expressions.leftEyeClosed && !expressions.rightEyeClosed) {
      detectedExpressions.push({ text: 'Left Wink', color: '#FF69B4', intensity: 1 });
    } else if (expressions.rightEyeClosed && !expressions.leftEyeClosed) {
      detectedExpressions.push({ text: 'Right Wink', color: '#FF69B4', intensity: 1 });
    }
    if (expressions.cheeksPuffed) {
      detectedExpressions.push({ text: 'Cheeks Puffed', color: '#98FB98', intensity: 1 });
    }
    if (expressions.mouthPucker) {
      detectedExpressions.push({ text: 'Puckering', color: '#FF1493', intensity: 1 });
    }
    if (expressions.leftEyebrowRaised || expressions.rightEyebrowRaised) {
      const side = expressions.leftEyebrowRaised && expressions.rightEyebrowRaised ? '' : 
                   expressions.leftEyebrowRaised ? 'Left ' : 'Right ';
      detectedExpressions.push({ text: `${side}Brow Raised`, color: '#00CED1', intensity: expressions.browRaiseAmount });
    }

    // Draw expression labels in top right corner
    if (detectedExpressions.length > 0) {
      const startX = canvasWidth - 20;
      const startY = 25;
      const lineHeight = 28;

      this.ctx.textAlign = 'right';
      this.ctx.textBaseline = 'top';

      detectedExpressions.forEach((expr, index) => {
        const y = startY + index * lineHeight;
        
        // Background pill
        this.ctx.font = 'bold 14px sans-serif';
        const textWidth = this.ctx.measureText(expr.text).width;
        
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        this.ctx.beginPath();
        this.ctx.roundRect(startX - textWidth - 16, y - 4, textWidth + 20, 24, 12);
        this.ctx.fill();
        
        // Colored indicator dot
        this.ctx.fillStyle = expr.color;
        this.ctx.beginPath();
        this.ctx.arc(startX - textWidth - 8, y + 8, 4, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Text
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillText(expr.text, startX - 8, y);
      });
    } else {
      // Show "Neutral" when no expressions detected
      this.ctx.textAlign = 'right';
      this.ctx.textBaseline = 'top';
      this.ctx.font = 'bold 14px sans-serif';
      
      const text = 'Neutral';
      const textWidth = this.ctx.measureText(text).width;
      
      this.ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
      this.ctx.beginPath();
      this.ctx.roundRect(canvasWidth - textWidth - 36, 21, textWidth + 20, 24, 12);
      this.ctx.fill();
      
      this.ctx.fillStyle = '#888888';
      this.ctx.beginPath();
      this.ctx.arc(canvasWidth - textWidth - 28, 33, 4, 0, Math.PI * 2);
      this.ctx.fill();
      
      this.ctx.fillStyle = '#aaaaaa';
      this.ctx.fillText(text, canvasWidth - 28, 25);
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

  // Draw custom connections between landmarks
  drawCustomConnections(landmarks, canvasWidth, canvasHeight, connections) {
    if (!landmarks || !connections || connections.length === 0) return;
    
    this.ctx.save();
    this.ctx.strokeStyle = '#ff00ff';
    this.ctx.lineWidth = 3;
    this.ctx.lineCap = 'round';
    this.ctx.shadowColor = '#ff00ff';
    this.ctx.shadowBlur = 8;
    
    for (const [idx1, idx2] of connections) {
      if (idx1 >= landmarks.length || idx2 >= landmarks.length) continue;
      
      const p1 = landmarks[idx1];
      const p2 = landmarks[idx2];
      
      const x1 = p1.x * canvasWidth;
      const y1 = p1.y * canvasHeight;
      const x2 = p2.x * canvasWidth;
      const y2 = p2.y * canvasHeight;
      
      this.ctx.beginPath();
      this.ctx.moveTo(x1, y1);
      this.ctx.lineTo(x2, y2);
      this.ctx.stroke();
      
      // Draw small circles at endpoints
      this.ctx.fillStyle = '#ff00ff';
      this.ctx.beginPath();
      this.ctx.arc(x1, y1, 4, 0, Math.PI * 2);
      this.ctx.fill();
      this.ctx.beginPath();
      this.ctx.arc(x2, y2, 4, 0, Math.PI * 2);
      this.ctx.fill();
    }
    
    this.ctx.restore();
  }

  // Highlight a selected landmark
  highlightLandmark(landmarks, canvasWidth, canvasHeight, landmarkIdx) {
    if (!landmarks || landmarkIdx >= landmarks.length) return;
    
    const lm = landmarks[landmarkIdx];
    const x = lm.x * canvasWidth;
    const y = lm.y * canvasHeight;
    
    this.ctx.save();
    
    // Pulsing ring effect
    const time = performance.now() * 0.005;
    const pulseSize = 12 + Math.sin(time) * 4;
    
    // Outer glow
    this.ctx.strokeStyle = '#ffff00';
    this.ctx.lineWidth = 3;
    this.ctx.shadowColor = '#ffff00';
    this.ctx.shadowBlur = 15;
    
    this.ctx.beginPath();
    this.ctx.arc(x, y, pulseSize, 0, Math.PI * 2);
    this.ctx.stroke();
    
    // Inner filled circle
    this.ctx.fillStyle = 'rgba(255, 255, 0, 0.5)';
    this.ctx.beginPath();
    this.ctx.arc(x, y, 6, 0, Math.PI * 2);
    this.ctx.fill();
    
    // Label
    this.ctx.shadowBlur = 0;
    this.ctx.fillStyle = '#ffffff';
    this.ctx.font = 'bold 12px sans-serif';
    this.ctx.fillText(`#${landmarkIdx}`, x + 15, y - 10);
    
    this.ctx.restore();
  }

  /**
   * Draw mood indicator with valence-arousal plot
   * @param {Object} moodData - Data from MoodAnalyzer.analyze()
   * @param {number} canvasWidth - Canvas width
   * @param {number} canvasHeight - Canvas height
   * @param {Object} options - Display options
   */
  drawMoodIndicator(moodData, canvasWidth, canvasHeight, options = {}) {
    if (!moodData) return;

    const {
      showMoodLabel = true,
      showEmotionWheel = true,
      showEmotionBars = false,
      position = 'top-right'
    } = options;

    const { mood, valence, arousal, dominantEmotion, emotions, confidence } = moodData;

    this.ctx.save();

    // Mood colors based on valence-arousal
    const getMoodColor = (v, a) => {
      const hue = ((v + 1) / 2) * 120;
      const saturation = 60 + Math.abs(a) * 40;
      const lightness = 40 + (a + 1) * 15;
      return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    };

    const moodColor = getMoodColor(valence, arousal);

    // Position calculations
    let startX, startY;
    const isLeftSide = position === 'top-left';
    
    if (position === 'top-right') {
      startX = canvasWidth - 20;
      startY = 20;
    } else if (position === 'top-left') {
      startX = 20;
      startY = 20;
    } else {
      startX = canvasWidth - 20;
      startY = 20;
    }

    let currentY = startY;

    // Draw mood label
    if (showMoodLabel) {
      this.ctx.textBaseline = 'top';
      this.ctx.font = 'bold 18px sans-serif';
      
      const labelText = mood;
      const textWidth = this.ctx.measureText(labelText).width;
      const pillWidth = textWidth + 32;
      
      if (isLeftSide) {
        // Left-aligned layout
        this.ctx.textAlign = 'left';
        
        // Background pill
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.beginPath();
        this.ctx.roundRect(startX, currentY - 4, pillWidth, 30, 15);
        this.ctx.fill();
        
        // Mood indicator circle
        this.ctx.fillStyle = moodColor;
        this.ctx.beginPath();
        this.ctx.arc(startX + 14, currentY + 11, 6, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Mood text
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillText(labelText, startX + 26, currentY);
        
        // Confidence bar under the label
        const barWidth = textWidth + 16;
        const barHeight = 3;
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        this.ctx.fillRect(startX + 8, currentY + 26, barWidth, barHeight);
        this.ctx.fillStyle = moodColor;
        this.ctx.fillRect(startX + 8, currentY + 26, barWidth * confidence, barHeight);
      } else {
        // Right-aligned layout (original)
        this.ctx.textAlign = 'right';
        
        // Background pill
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.beginPath();
        this.ctx.roundRect(startX - textWidth - 24, currentY - 4, pillWidth, 30, 15);
        this.ctx.fill();
        
        // Mood indicator circle
        this.ctx.fillStyle = moodColor;
        this.ctx.beginPath();
        this.ctx.arc(startX - textWidth - 14, currentY + 11, 6, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Mood text
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillText(labelText, startX - 8, currentY);
        
        // Confidence bar under the label
        const barWidth = textWidth + 16;
        const barHeight = 3;
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        this.ctx.fillRect(startX - barWidth - 8, currentY + 26, barWidth, barHeight);
        this.ctx.fillStyle = moodColor;
        this.ctx.fillRect(startX - barWidth - 8, currentY + 26, barWidth * confidence, barHeight);
      }
      
      currentY += 40;
    }

    // Draw valence-arousal wheel/plot
    if (showEmotionWheel) {
      const wheelSize = 80;
      const wheelX = isLeftSide ? startX + wheelSize / 2 + 10 : startX - wheelSize / 2 - 10;
      const wheelY = currentY + wheelSize / 2 + 10;
      
      // Background circle
      this.ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      this.ctx.beginPath();
      this.ctx.arc(wheelX, wheelY, wheelSize / 2 + 8, 0, Math.PI * 2);
      this.ctx.fill();
      
      // Quadrant colors (subtle)
      const quadrantColors = [
        { color: 'rgba(100, 255, 100, 0.15)', startAngle: -Math.PI / 2, endAngle: 0 },
        { color: 'rgba(255, 100, 100, 0.15)', startAngle: -Math.PI, endAngle: -Math.PI / 2 },
        { color: 'rgba(100, 100, 255, 0.15)', startAngle: Math.PI / 2, endAngle: Math.PI },
        { color: 'rgba(100, 255, 255, 0.15)', startAngle: 0, endAngle: Math.PI / 2 }
      ];
      
      for (const q of quadrantColors) {
        this.ctx.fillStyle = q.color;
        this.ctx.beginPath();
        this.ctx.moveTo(wheelX, wheelY);
        this.ctx.arc(wheelX, wheelY, wheelSize / 2, q.startAngle, q.endAngle);
        this.ctx.closePath();
        this.ctx.fill();
      }
      
      // Axis lines
      this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
      this.ctx.lineWidth = 1;
      this.ctx.beginPath();
      this.ctx.moveTo(wheelX - wheelSize / 2, wheelY);
      this.ctx.lineTo(wheelX + wheelSize / 2, wheelY);
      this.ctx.moveTo(wheelX, wheelY - wheelSize / 2);
      this.ctx.lineTo(wheelX, wheelY + wheelSize / 2);
      this.ctx.stroke();
      
      // Axis labels
      this.ctx.font = '9px sans-serif';
      this.ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText('−', wheelX - wheelSize / 2 - 6, wheelY);
      this.ctx.fillText('+', wheelX + wheelSize / 2 + 6, wheelY);
      this.ctx.fillText('↑', wheelX, wheelY - wheelSize / 2 - 6);
      this.ctx.fillText('↓', wheelX, wheelY + wheelSize / 2 + 6);
      
      // Outer ring
      this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      this.ctx.lineWidth = 2;
      this.ctx.beginPath();
      this.ctx.arc(wheelX, wheelY, wheelSize / 2, 0, Math.PI * 2);
      this.ctx.stroke();
      
      // Current position dot
      const dotX = wheelX + (valence * wheelSize / 2);
      const dotY = wheelY - (arousal * wheelSize / 2); // Invert Y for screen coords
      
      // Glow effect
      this.ctx.shadowColor = moodColor;
      this.ctx.shadowBlur = 10;
      
      // Dot
      this.ctx.fillStyle = moodColor;
      this.ctx.beginPath();
      this.ctx.arc(dotX, dotY, 8, 0, Math.PI * 2);
      this.ctx.fill();
      
      // Inner highlight
      this.ctx.shadowBlur = 0;
      this.ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
      this.ctx.beginPath();
      this.ctx.arc(dotX - 2, dotY - 2, 3, 0, Math.PI * 2);
      this.ctx.fill();
      
      // Quadrant labels
      this.ctx.font = '8px sans-serif';
      this.ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      this.ctx.textAlign = 'center';
      this.ctx.fillText('Happy', wheelX + wheelSize / 4, wheelY - wheelSize / 4);
      this.ctx.fillText('Angry', wheelX - wheelSize / 4, wheelY - wheelSize / 4);
      this.ctx.fillText('Sad', wheelX - wheelSize / 4, wheelY + wheelSize / 4);
      this.ctx.fillText('Calm', wheelX + wheelSize / 4, wheelY + wheelSize / 4);
      
      currentY += wheelSize + 30;
    }

    // Draw emotion bars
    if (showEmotionBars && emotions) {
      const barWidth = 100;
      const barHeight = 8;
      const barSpacing = 16;
      
      const emotionList = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust'];
      const emotionColors = {
        happy: '#FFD700',
        sad: '#6495ED',
        angry: '#FF4444',
        fear: '#9370DB',
        surprise: '#00CED1',
        disgust: '#98FB98'
      };
      
      this.ctx.font = '10px sans-serif';
      
      for (let i = 0; i < emotionList.length; i++) {
        const emotion = emotionList[i];
        const value = emotions[emotion] || 0;
        const y = currentY + i * barSpacing;
        const emotionLabel = emotion.charAt(0).toUpperCase() + emotion.slice(1);
        
        if (isLeftSide) {
          // Left-aligned layout
          this.ctx.textAlign = 'left';
          this.ctx.fillStyle = '#ffffff';
          this.ctx.fillText(emotionLabel, startX, y + 6);
          
          // Background bar (after label)
          const labelWidth = this.ctx.measureText(emotionLabel).width;
          this.ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
          this.ctx.fillRect(startX + labelWidth + 10, y, barWidth, barHeight);
          
          // Value bar
          this.ctx.fillStyle = emotionColors[emotion];
          this.ctx.fillRect(startX + labelWidth + 10, y, barWidth * Math.min(1, value * 2), barHeight);
        } else {
          // Right-aligned layout
          this.ctx.textAlign = 'right';
          this.ctx.fillStyle = '#ffffff';
          this.ctx.fillText(emotionLabel, startX - barWidth - 10, y + 6);
          
          // Background bar
          this.ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
          this.ctx.fillRect(startX - barWidth - 5, y, barWidth, barHeight);
          
          // Value bar
          this.ctx.fillStyle = emotionColors[emotion];
          this.ctx.fillRect(startX - barWidth - 5, y, barWidth * Math.min(1, value * 2), barHeight);
        }
      }
    }

    this.ctx.restore();
  }

  /**
   * Draw hand landmarks and connections
   * @param {Object} handData - Hand tracking results from HandTracker
   * @param {number} canvasWidth - Canvas width
   * @param {number} canvasHeight - Canvas height
   * @param {Object} options - Drawing options
   */
  drawHands(handData, canvasWidth, canvasHeight, options = {}) {
    if (!handData || !handData.hands || handData.hands.length === 0) return;

    const {
      showConnections = true,
      showLandmarks = true,
      showLabels = true,
      showGesture = true,
      connectionColor = '#00ff88',
      landmarkColor = '#ffffff',
      leftHandColor = '#ff6b6b',
      rightHandColor = '#4ecdc4',
      lineWidth = 3,
      landmarkRadius = 5,
      mirrorMode = true  // Match mirrored video display
    } = options;

    this.ctx.save();

    for (const hand of handData.hands) {
      const { landmarks, handedness, confidence } = hand;
      if (!landmarks || landmarks.length < 21) continue;

      // Handedness is based on actual hand anatomy, not screen position
      // So we don't swap the label - just mirror the coordinates
      const displayHandedness = handedness;
      
      // Choose color based on handedness
      const isLeft = handedness === 'Left';
      const handColor = isLeft ? leftHandColor : rightHandColor;
      
      // Helper to get mirrored X coordinate
      const getMirroredX = (x) => mirrorMode ? (1 - x) : x;

      // Draw connections (skeleton)
      if (showConnections) {
        this.ctx.strokeStyle = handColor;
        this.ctx.lineWidth = lineWidth;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        
        // Add glow effect
        this.ctx.shadowColor = handColor;
        this.ctx.shadowBlur = 8;

        for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
          const start = landmarks[startIdx];
          const end = landmarks[endIdx];
          
          if (!start || !end) continue;

          const x1 = getMirroredX(start.x) * canvasWidth;
          const y1 = start.y * canvasHeight;
          const x2 = getMirroredX(end.x) * canvasWidth;
          const y2 = end.y * canvasHeight;

          this.ctx.beginPath();
          this.ctx.moveTo(x1, y1);
          this.ctx.lineTo(x2, y2);
          this.ctx.stroke();
        }
        
        this.ctx.shadowBlur = 0;
      }

      // Draw landmarks
      if (showLandmarks) {
        for (let i = 0; i < landmarks.length; i++) {
          const lm = landmarks[i];
          const x = getMirroredX(lm.x) * canvasWidth;
          const y = lm.y * canvasHeight;

          // Fingertips get larger dots
          const isTip = [4, 8, 12, 16, 20].includes(i);
          const radius = isTip ? landmarkRadius * 1.5 : landmarkRadius;

          // Draw landmark dot
          this.ctx.beginPath();
          this.ctx.arc(x, y, radius, 0, Math.PI * 2);
          
          // Gradient fill for 3D effect
          const gradient = this.ctx.createRadialGradient(x - radius/3, y - radius/3, 0, x, y, radius);
          gradient.addColorStop(0, '#ffffff');
          gradient.addColorStop(0.5, handColor);
          gradient.addColorStop(1, handColor);
          
          this.ctx.fillStyle = isTip ? '#ffffff' : gradient;
          this.ctx.fill();
          
          // Border
          this.ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
          this.ctx.lineWidth = 1;
          this.ctx.stroke();
        }
      }

      // Draw hand label
      if (showLabels) {
        const wrist = landmarks[0];
        const x = getMirroredX(wrist.x) * canvasWidth;
        const y = wrist.y * canvasHeight + 30;

        const label = `${displayHandedness} (${Math.round(confidence * 100)}%)`;
        
        this.ctx.font = 'bold 14px sans-serif';
        const textWidth = this.ctx.measureText(label).width;
        
        // Background
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.beginPath();
        this.ctx.roundRect(x - textWidth/2 - 8, y - 12, textWidth + 16, 24, 8);
        this.ctx.fill();
        
        // Text
        this.ctx.fillStyle = handColor;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(label, x, y);
      }

      // Detect and show gesture
      if (showGesture) {
        const gesture = HandTracker.detectGesture(landmarks);
        if (gesture && gesture !== 'unknown' && gesture !== 'partial') {
          const wrist = landmarks[0];
          const x = getMirroredX(wrist.x) * canvasWidth;
          const y = wrist.y * canvasHeight - 20;

          const gestureLabel = gesture.replace('_', ' ').toUpperCase();
          
          this.ctx.font = 'bold 16px sans-serif';
          const textWidth = this.ctx.measureText(gestureLabel).width;
          
          // Background pill
          this.ctx.fillStyle = handColor;
          this.ctx.beginPath();
          this.ctx.roundRect(x - textWidth/2 - 10, y - 14, textWidth + 20, 28, 14);
          this.ctx.fill();
          
          // Text
          this.ctx.fillStyle = '#000000';
          this.ctx.textAlign = 'center';
          this.ctx.textBaseline = 'middle';
          this.ctx.fillText(gestureLabel, x, y);
        }
      }
    }

    this.ctx.restore();
  }
}
