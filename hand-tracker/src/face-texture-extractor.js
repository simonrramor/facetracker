// Face Texture Extractor
// Extracts face texture from uploaded images and warps to UV layout

import { FACE_MESH_TRIANGULATION, FACE_MESH_UVS } from './face-mesh-data.js';

export class FaceTextureExtractor {
  constructor() {
    this.faceMesh = null;
    this.initialized = false;
    this.outputSize = 512; // Output texture size
  }

  async initialize() {
    if (this.initialized) return;
    
    console.log('FaceTextureExtractor: Starting initialization...');
    
    // Wait for FaceMesh to be available from CDN
    if (typeof window.FaceMesh === 'undefined') {
      throw new Error('FaceMesh not loaded. Make sure the CDN script is included.');
    }
    
    console.log('FaceTextureExtractor: Creating FaceMesh instance...');
    this.faceMesh = new window.FaceMesh({
      locateFile: (file) => {
        console.log('FaceTextureExtractor: Loading file:', file);
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
      }
    });

    console.log('FaceTextureExtractor: Setting options...');
    this.faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    // Set up a pending results handler
    this.pendingResolve = null;
    this.pendingReject = null;
    
    this.faceMesh.onResults((results) => {
      console.log('FaceTextureExtractor: Got results, faces:', results.multiFaceLandmarks?.length || 0);
      if (this.pendingResolve) {
        const resolve = this.pendingResolve;
        const reject = this.pendingReject;
        this.pendingResolve = null;
        this.pendingReject = null;
        
        if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
          reject(new Error('No face detected in image'));
          return;
        }
        
        resolve(results.multiFaceLandmarks[0]);
      }
    });

    // Initialize the model by calling initialize() if available
    if (this.faceMesh.initialize) {
      console.log('FaceTextureExtractor: Calling faceMesh.initialize()...');
      await this.faceMesh.initialize();
      console.log('FaceTextureExtractor: faceMesh.initialize() complete');
    }

    this.initialized = true;
    console.log('FaceTextureExtractor initialized successfully');
  }

  // Extract face texture from an image
  async extractTexture(imageElement) {
    console.log('FaceTextureExtractor: Extracting texture from image', imageElement.width, 'x', imageElement.height);
    
    // Get landmarks from the image with timeout
    const landmarks = await new Promise((resolve, reject) => {
      this.pendingResolve = resolve;
      this.pendingReject = reject;
      
      // Set a timeout in case FaceMesh doesn't respond
      const timeout = setTimeout(() => {
        if (this.pendingResolve) {
          this.pendingResolve = null;
          this.pendingReject = null;
          reject(new Error('Face detection timed out'));
        }
      }, 30000); // 30 second timeout
      
      // Wrap resolve to clear timeout
      const originalResolve = resolve;
      this.pendingResolve = (result) => {
        clearTimeout(timeout);
        originalResolve(result);
      };
      
      const originalReject = reject;
      this.pendingReject = (error) => {
        clearTimeout(timeout);
        originalReject(error);
      };
      
      // Send the image for processing
      console.log('FaceTextureExtractor: Sending image to FaceMesh...');
      this.faceMesh.send({ image: imageElement });
    });
    
    console.log('FaceTextureExtractor: Got', landmarks.length, 'landmarks, warping to UV...');
    const texture = this.warpFaceToUV(imageElement, landmarks);
    console.log('FaceTextureExtractor: Texture created');
    
    // Return both texture and landmarks for head pose detection
    return { texture, landmarks };
  }

  // Detect head pose from landmarks (yaw angle)
  detectHeadPose(landmarks) {
    if (!landmarks || landmarks.length < 468) return 'front';
    
    // Use eye positions and nose to estimate yaw
    const leftEyeOuter = landmarks[33];
    const rightEyeOuter = landmarks[263];
    const noseTip = landmarks[1];
    
    // Also check face contour asymmetry for better detection
    const leftContour = landmarks[234];  // Left face edge
    const rightContour = landmarks[454]; // Right face edge
    
    // Face width based on eyes
    const eyeDistance = rightEyeOuter.x - leftEyeOuter.x;
    
    // Nose position relative to eye center
    const eyeCenter = (leftEyeOuter.x + rightEyeOuter.x) / 2;
    const noseOffset = noseTip.x - eyeCenter;
    
    // Also check contour asymmetry (more reliable for larger angles)
    const leftEdgeDist = eyeCenter - leftContour.x;
    const rightEdgeDist = rightContour.x - eyeCenter;
    const contourRatio = leftEdgeDist / (rightEdgeDist + 0.001);
    
    // Normalize by face width
    const normalizedOffset = noseOffset / eyeDistance;
    
    // Use both metrics - lower threshold (0.03) for more sensitivity
    // When person looks RIGHT: nose appears left, left side of face more visible
    // When person looks LEFT: nose appears right, right side of face more visible
    if (normalizedOffset < -0.03 || contourRatio > 1.15) {
      return 'right'; // Looking right (their right), left side of face visible
    }
    if (normalizedOffset > 0.03 || contourRatio < 0.85) {
      return 'left';  // Looking left (their left), right side of face visible
    }
    return 'front';
  }

  // Extract face and warp to canonical UV layout
  // Uses 3-point affine transform to align eyes and nose consistently
  warpFaceToUV(sourceImage, landmarks) {
    const size = this.outputSize;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    const srcWidth = sourceImage.width || sourceImage.videoWidth;
    const srcHeight = sourceImage.height || sourceImage.videoHeight;

    // Fill with opaque background first (will be masked by face shape)
    ctx.fillStyle = 'rgba(128, 128, 128, 0)';
    ctx.fillRect(0, 0, size, size);

    // Key anchor points for alignment
    const leftEye = landmarks[33];   // Left eye outer corner
    const rightEye = landmarks[263]; // Right eye outer corner
    const noseTip = landmarks[1];    // Nose tip
    
    // Source positions (in pixels)
    const srcLeftEye = { x: leftEye.x * srcWidth, y: leftEye.y * srcHeight };
    const srcRightEye = { x: rightEye.x * srcWidth, y: rightEye.y * srcHeight };
    const srcNose = { x: noseTip.x * srcWidth, y: noseTip.y * srcHeight };
    
    // Canonical destination positions (fixed layout in texture space)
    // These MUST match the UV coordinates used in webgl-mask-renderer.js
    const dstLeftEye = { x: size * 0.30, y: size * 0.35 };
    const dstRightEye = { x: size * 0.70, y: size * 0.35 };
    const dstNose = { x: size * 0.50, y: size * 0.55 };
    
    // Compute affine transform from source to destination using 3 points
    const transform = this.computeAffineTransform(
      srcLeftEye, srcRightEye, srcNose,
      dstLeftEye, dstRightEye, dstNose
    );
    
    if (!transform) {
      return this.simpleBoundingBoxExtract(sourceImage, landmarks, size);
    }
    
    ctx.save();
    
    // Apply the affine transform and draw the ENTIRE image
    // No clipping - let the full face be drawn, shader handles edges
    ctx.setTransform(
      transform.a, transform.d,  // scale/rotate
      transform.b, transform.e,  // skew
      transform.c, transform.f   // translate
    );
    
    ctx.drawImage(sourceImage, 0, 0);
    
    ctx.restore();
    
    // Now apply a soft face mask to fade edges
    this.applyFaceMask(ctx, landmarks, transform, srcWidth, srcHeight, size);
    
    return canvas;
  }
  
  // Apply a soft mask based on face outline
  applyFaceMask(ctx, landmarks, transform, srcWidth, srcHeight, size) {
    // Create a mask canvas
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = size;
    maskCanvas.height = size;
    const maskCtx = maskCanvas.getContext('2d');
    
    // Draw face silhouette as white on black
    const silhouette = [
      10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
      397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
      172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ];
    
    const transformPoint = (srcX, srcY) => ({
      x: transform.a * srcX + transform.b * srcY + transform.c,
      y: transform.d * srcX + transform.e * srcY + transform.f
    });
    
    maskCtx.fillStyle = 'black';
    maskCtx.fillRect(0, 0, size, size);
    
    maskCtx.beginPath();
    const firstLm = landmarks[silhouette[0]];
    const firstPt = transformPoint(firstLm.x * srcWidth, firstLm.y * srcHeight);
    maskCtx.moveTo(firstPt.x, firstPt.y);
    
    for (let i = 1; i < silhouette.length; i++) {
      const lm = landmarks[silhouette[i]];
      if (lm) {
        const pt = transformPoint(lm.x * srcWidth, lm.y * srcHeight);
        maskCtx.lineTo(pt.x, pt.y);
      }
    }
    maskCtx.closePath();
    maskCtx.fillStyle = 'white';
    maskCtx.fill();
    
    // Apply mask using destination-in compositing
    ctx.globalCompositeOperation = 'destination-in';
    ctx.drawImage(maskCanvas, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
  }
  
  // Compute affine transform from 3 source points to 3 destination points
  computeAffineTransform(s0, s1, s2, d0, d1, d2) {
    // Solve for transform matrix [a b c; d e f]
    // where: dx = a*sx + b*sy + c, dy = d*sx + e*sy + f
    
    const det = (s0.x - s2.x) * (s1.y - s2.y) - (s1.x - s2.x) * (s0.y - s2.y);
    if (Math.abs(det) < 1e-10) return null;
    
    const a = ((d0.x - d2.x) * (s1.y - s2.y) - (d1.x - d2.x) * (s0.y - s2.y)) / det;
    const b = ((d1.x - d2.x) * (s0.x - s2.x) - (d0.x - d2.x) * (s1.x - s2.x)) / det;
    const c = d2.x - a * s2.x - b * s2.y;
    
    const d = ((d0.y - d2.y) * (s1.y - s2.y) - (d1.y - d2.y) * (s0.y - s2.y)) / det;
    const e = ((d1.y - d2.y) * (s0.x - s2.x) - (d0.y - d2.y) * (s1.x - s2.x)) / det;
    const f = d2.y - d * s2.x - e * s2.y;
    
    return { a, b, c, d, e, f };
  }
  
  // Fallback simple extraction
  simpleBoundingBoxExtract(sourceImage, landmarks, size) {
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    
    const srcWidth = sourceImage.width || sourceImage.videoWidth;
    const srcHeight = sourceImage.height || sourceImage.videoHeight;
    
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < Math.min(468, landmarks.length); i++) {
      minX = Math.min(minX, landmarks[i].x * srcWidth);
      minY = Math.min(minY, landmarks[i].y * srcHeight);
      maxX = Math.max(maxX, landmarks[i].x * srcWidth);
      maxY = Math.max(maxY, landmarks[i].y * srcHeight);
    }
    
    const padding = 20;
    minX = Math.max(0, minX - padding);
    minY = Math.max(0, minY - padding);
    maxX = Math.min(srcWidth, maxX + padding);
    maxY = Math.min(srcHeight, maxY + padding);
    
    ctx.drawImage(
      sourceImage,
      minX, minY, maxX - minX, maxY - minY,
      0, 0, size, size
    );
    
    return canvas;
  }

  // Apply proper UV mapping using triangles
  applyUVMapping(ctx, sourceImage, landmarks, srcWidth, srcHeight, size) {
    const triangles = FACE_MESH_TRIANGULATION;
    const uvs = FACE_MESH_UVS;
    
    // Clear and redraw with proper UV mapping
    ctx.clearRect(0, 0, size, size);

    for (let i = 0; i < triangles.length; i += 3) {
      const idx0 = triangles[i];
      const idx1 = triangles[i + 1];
      const idx2 = triangles[i + 2];

      // Skip if any landmark or UV is invalid
      if (!landmarks[idx0] || !landmarks[idx1] || !landmarks[idx2]) continue;
      if (idx0 >= 468 || idx1 >= 468 || idx2 >= 468) continue;

      // Source triangle (from image)
      const src0 = { x: landmarks[idx0].x * srcWidth, y: landmarks[idx0].y * srcHeight };
      const src1 = { x: landmarks[idx1].x * srcWidth, y: landmarks[idx1].y * srcHeight };
      const src2 = { x: landmarks[idx2].x * srcWidth, y: landmarks[idx2].y * srcHeight };

      // Destination triangle (UV space)
      // Note: UV coordinates are mirrored compared to MediaPipe landmarks
      // Need to flip X to match orientation, and flip Y for WebGL texture coords
      const dst0 = { x: (1.0 - uvs[idx0 * 2]) * size, y: (1.0 - uvs[idx0 * 2 + 1]) * size };
      const dst1 = { x: (1.0 - uvs[idx1 * 2]) * size, y: (1.0 - uvs[idx1 * 2 + 1]) * size };
      const dst2 = { x: (1.0 - uvs[idx2 * 2]) * size, y: (1.0 - uvs[idx2 * 2 + 1]) * size };

      this.drawWarpedTriangle(ctx, sourceImage, src0, src1, src2, dst0, dst1, dst2);
    }
  }

  // Draw a single warped triangle using affine transformation
  drawWarpedTriangle(ctx, img, s0, s1, s2, d0, d1, d2) {
    // Use a different approach: compute inverse transform and iterate destination pixels
    // This is more reliable than setTransform for texture mapping
    
    // Compute dest->source transform (inverse)
    const det = (d0.x - d2.x) * (d1.y - d2.y) - (d1.x - d2.x) * (d0.y - d2.y);
    if (Math.abs(det) < 1e-10) return;
    
    const a = ((s0.x - s2.x) * (d1.y - d2.y) - (s1.x - s2.x) * (d0.y - d2.y)) / det;
    const b = ((s1.x - s2.x) * (d0.x - d2.x) - (s0.x - s2.x) * (d1.x - d2.x)) / det;
    const c = s2.x - a * d2.x - b * d2.y;
    const d = ((s0.y - s2.y) * (d1.y - d2.y) - (s1.y - s2.y) * (d0.y - d2.y)) / det;
    const e = ((s1.y - s2.y) * (d0.x - d2.x) - (s0.y - s2.y) * (d1.x - d2.x)) / det;
    const f = s2.y - d * d2.x - e * d2.y;

    ctx.save();
    
    // Expand destination triangle slightly to eliminate seams
    const expand = 1.0;
    const cx = (d0.x + d1.x + d2.x) / 3;
    const cy = (d0.y + d1.y + d2.y) / 3;
    const expandPt = (p) => ({
      x: cx + (p.x - cx) * (1 + expand / Math.max(10, Math.hypot(p.x - cx, p.y - cy))),
      y: cy + (p.y - cy) * (1 + expand / Math.max(10, Math.hypot(p.x - cx, p.y - cy)))
    });
    const ed0 = expandPt(d0), ed1 = expandPt(d1), ed2 = expandPt(d2);
    
    // Clip to destination triangle
    ctx.beginPath();
    ctx.moveTo(ed0.x, ed0.y);
    ctx.lineTo(ed1.x, ed1.y);
    ctx.lineTo(ed2.x, ed2.y);
    ctx.closePath();
    ctx.clip();
    
    // Now use setTransform with the INVERSE (dest->src) transform
    // When we draw the destination rectangle, canvas will sample from source
    // We need to "pretend" the image is in destination space, then transform to source
    // Actually, use the forward transform and draw image
    
    // Compute forward transform (src->dest)
    const detFwd = (s0.x - s2.x) * (s1.y - s2.y) - (s1.x - s2.x) * (s0.y - s2.y);
    const af = ((d0.x - d2.x) * (s1.y - s2.y) - (d1.x - d2.x) * (s0.y - s2.y)) / detFwd;
    const bf = ((d1.x - d2.x) * (s0.x - s2.x) - (d0.x - d2.x) * (s1.x - s2.x)) / detFwd;
    const cf = d2.x - af * s2.x - bf * s2.y;
    const df = ((d0.y - d2.y) * (s1.y - s2.y) - (d1.y - d2.y) * (s0.y - s2.y)) / detFwd;
    const ef = ((d1.y - d2.y) * (s0.x - s2.x) - (d0.y - d2.y) * (s1.x - s2.x)) / detFwd;
    const ff = d2.y - df * s2.x - ef * s2.y;
    
    // setTransform(a, b, c, d, e, f) where:
    // x' = a*x + c*y + e
    // y' = b*x + d*y + f
    // Our forward: x' = af*x + bf*y + cf, y' = df*x + ef*y + ff
    // So: ctx.setTransform(af, df, bf, ef, cf, ff)
    ctx.setTransform(af, df, bf, ef, cf, ff);
    
    const srcWidth = img.width || img.videoWidth;
    const srcHeight = img.height || img.videoHeight;
    ctx.drawImage(img, 0, 0, srcWidth, srcHeight);
    
    ctx.restore();
  }

  // Solve for affine transformation matrix
  // Returns transform that maps SOURCE to DESTINATION
  solveAffine(s0, s1, s2, d0, d1, d2) {
    // Calculate the affine transformation from SOURCE to DESTINATION
    const det = (s0.x - s2.x) * (s1.y - s2.y) - (s1.x - s2.x) * (s0.y - s2.y);
    
    if (Math.abs(det) < 1e-10) {
      return null; // Degenerate triangle
    }

    const a = ((d0.x - d2.x) * (s1.y - s2.y) - (d1.x - d2.x) * (s0.y - s2.y)) / det;
    const b = ((d1.x - d2.x) * (s0.x - s2.x) - (d0.x - d2.x) * (s1.x - s2.x)) / det;
    const c = d2.x - a * s2.x - b * s2.y;

    const d = ((d0.y - d2.y) * (s1.y - s2.y) - (d1.y - d2.y) * (s0.y - s2.y)) / det;
    const e = ((d1.y - d2.y) * (s0.x - s2.x) - (d0.y - d2.y) * (s1.x - s2.x)) / det;
    const f = d2.y - d * s2.x - e * s2.y;

    return { a, b, c, d, e, f };
  }

  // Get the extracted texture as an ImageData or canvas
  getTextureCanvas() {
    return this.outputCanvas;
  }
}

// Helper to load an image from a file
export function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = e.target.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Helper to load an image from URL
export function loadImageFromURL(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}
