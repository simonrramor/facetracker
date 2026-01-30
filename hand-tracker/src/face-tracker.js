// Face Tracker using MediaPipe Face Mesh
// Provides 468 facial landmarks for detailed face tracking

// MediaPipe FaceMesh will be loaded from CDN and available as window.FaceMesh

export class FaceTracker {
  constructor(videoElement, onResults) {
    this.video = videoElement;
    this.onResults = onResults;
    this.faceMesh = null;
    this.running = false;
    this.animationId = null;
  }

  async initialize() {
    // Wait for FaceMesh to be available (loaded from CDN in HTML)
    if (typeof window.FaceMesh === 'undefined') {
      throw new Error('FaceMesh not loaded. Make sure the CDN script is included.');
    }
    
    // Create FaceMesh instance
    this.faceMesh = new window.FaceMesh({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
      }
    });

    // Configure for detailed tracking
    this.faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,  // Enables iris tracking (478 total landmarks)
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    // Set up results callback
    this.faceMesh.onResults((results) => {
      if (this.onResults) {
        this.onResults(results);
      }
    });

    // Start the detection loop
    this.running = true;
    this.detectLoop();
  }

  async detectLoop() {
    if (!this.running) return;
    
    if (this.video.readyState >= 2) {
      try {
        await this.faceMesh.send({ image: this.video });
      } catch (e) {
        console.error('FaceMesh error:', e);
      }
    }
    
    this.animationId = requestAnimationFrame(() => this.detectLoop());
  }

  stop() {
    this.running = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
  }
}

// Landmark indices for key facial features
export const FACE_LANDMARKS = {
  // Face oval contour (36 points)
  silhouette: [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
  ],

  // Left eye (16 points)
  leftEye: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
  leftEyeCenter: 468,  // Iris center (with refineLandmarks)
  
  // Right eye (16 points)
  rightEye: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
  rightEyeCenter: 473,  // Iris center (with refineLandmarks)

  // Left eyebrow
  leftEyebrow: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],

  // Right eyebrow
  rightEyebrow: [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],

  // Nose
  noseBridge: [168, 6, 197, 195, 5],
  noseTip: 1,
  noseBottom: [2, 326, 327, 278, 279, 280, 281, 282, 283, 284],

  // Lips outer
  lipsOuter: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],

  // Lips inner
  lipsInner: [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191],

  // Key reference points for positioning
  forehead: 10,
  chin: 152,
  leftCheek: 234,
  rightCheek: 454
};

// Calculate face transform from landmarks
export function getFaceTransform(landmarks, canvasWidth, canvasHeight) {
  if (!landmarks || landmarks.length < 468) return null;

  // Convert normalized coordinates to pixels
  const toPixel = (lm) => ({
    x: lm.x * canvasWidth,
    y: lm.y * canvasHeight,
    z: lm.z * canvasWidth  // Z is also normalized to width
  });

  const leftEye = toPixel(landmarks[33]);
  const rightEye = toPixel(landmarks[263]);
  const nose = toPixel(landmarks[1]);
  const forehead = toPixel(landmarks[10]);
  const chin = toPixel(landmarks[152]);

  // Calculate eye distance (used for scaling)
  const eyeDistance = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y);

  // Calculate face angle (roll)
  const angle = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

  // Face center (between eyes, slightly below)
  const center = {
    x: (leftEye.x + rightEye.x) / 2,
    y: (leftEye.y + rightEye.y) / 2 + eyeDistance * 0.3
  };

  // Face dimensions
  const faceHeight = Math.hypot(chin.x - forehead.x, chin.y - forehead.y);
  const faceWidth = eyeDistance * 2.2;

  return {
    center,
    angle,
    eyeDistance,
    faceWidth,
    faceHeight,
    leftEye,
    rightEye,
    nose,
    forehead,
    chin,
    scale: eyeDistance / 100  // Normalized scale factor
  };
}
