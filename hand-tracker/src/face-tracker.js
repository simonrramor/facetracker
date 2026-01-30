// Face Tracker using MediaPipe Face Landmarker (tasks-vision)
// Provides 478 facial landmarks + 52 blendshapes for expressions

export class FaceTracker {
  constructor(videoElement, onResults, onProgress) {
    this.video = videoElement;
    this.onResults = onResults;
    this.onProgress = onProgress; // Progress callback: (percent, stage) => void
    this.faceLandmarker = null;
    this.running = false;
    this.animationId = null;
    this.lastVideoTime = -1;
  }

  // Report progress to callback
  reportProgress(percent, stage) {
    if (this.onProgress) {
      this.onProgress(percent, stage);
    }
  }

  // Fetch model with progress tracking
  async fetchModelWithProgress(url) {
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.status}`);
    }

    const contentLength = response.headers.get('content-length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;
    
    if (!total) {
      // If no content-length, just download without progress
      console.log('No content-length header, downloading without progress...');
      return await response.arrayBuffer();
    }

    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;

    while (true) {
      const { done, value } = await reader.read();
      
      if (done) break;
      
      chunks.push(value);
      received += value.length;
      
      const percent = Math.round((received / total) * 100);
      this.reportProgress(percent, 'model');
    }

    // Combine chunks into single ArrayBuffer
    const blob = new Blob(chunks);
    return await blob.arrayBuffer();
  }

  async initialize() {
    // Wait for the vision API to be loaded (promise set up in HTML)
    console.log('Waiting for Vision API...');
    this.reportProgress(0, 'init');
    
    const { FilesetResolver, FaceLandmarker } = await window.waitForVisionAPI;
    console.log('Vision API ready, initializing Face Landmarker...');
    
    try {
      // Initialize the vision tasks
      console.log('Loading WASM files...');
      this.reportProgress(5, 'wasm');
      
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );
      
      console.log('WASM loaded, downloading model...');
      this.reportProgress(15, 'model');

      // Model URL - using Google Storage (official)
      const modelUrl = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
      
      // Fetch model with progress tracking
      console.log('Downloading model with progress...');
      const modelBuffer = await this.fetchModelWithProgress(modelUrl);
      
      console.log('Model downloaded:', modelBuffer.byteLength, 'bytes');
      this.reportProgress(90, 'create');

      // Create Face Landmarker from the downloaded buffer
      console.log('Creating Face Landmarker...');
      const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetBuffer: new Uint8Array(modelBuffer)
        },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true
      });
      
      console.log('Face Landmarker created successfully');

      this.faceLandmarker = faceLandmarker;
      this.reportProgress(100, 'done');
      console.log('Face Landmarker initialized successfully');

      // Start the detection loop
      this.running = true;
      this.detectLoop();
    } catch (error) {
      console.error('Face Landmarker initialization failed:', error);
      throw error;
    }
  }

  detectLoop() {
    if (!this.running) return;

    if (this.video.readyState >= 2) {
      // For live webcam, always process each frame using performance.now() as timestamp
      const startTime = performance.now();
      
      try {
        const results = this.faceLandmarker.detectForVideo(this.video, startTime);
        
        if (this.onResults) {
          // Convert results to a format compatible with the old API
          // while also providing the new blendshapes data
          const convertedResults = this.convertResults(results);
          this.onResults(convertedResults);
        }
      } catch (e) {
        console.error('Face Landmarker error:', e);
      }
    }

    this.animationId = requestAnimationFrame(() => this.detectLoop());
  }

  convertResults(results) {
    // Convert new API results to be compatible with old format
    // while also exposing new features (blendshapes, transforms)
    const converted = {
      // Legacy format for backward compatibility
      multiFaceLandmarks: [],
      
      // New data
      faceLandmarks: results.faceLandmarks || [],
      faceBlendshapes: results.faceBlendshapes || [],
      facialTransformationMatrixes: results.facialTransformationMatrixes || [],
      
      // Processed expressions
      expressions: null
    };

    // Convert landmarks to old format (normalized coordinates with x, y, z)
    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
      converted.multiFaceLandmarks = results.faceLandmarks.map(landmarks => 
        landmarks.map(lm => ({
          x: lm.x,
          y: lm.y,
          z: lm.z
        }))
      );

      // Process blendshapes into easy-to-use expressions object
      if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
        converted.expressions = getExpressions(results.faceBlendshapes);
      }
    }

    return converted;
  }

  stop() {
    this.running = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
  }
}

// Process blendshapes into easy-to-use expressions
export function getExpressions(blendshapes) {
  if (!blendshapes || blendshapes.length === 0) return null;

  const scores = {};
  
  // Build scores object from blendshapes categories
  blendshapes[0]?.categories?.forEach(shape => {
    scores[shape.categoryName] = shape.score;
  });

  // Return processed expressions with thresholds applied
  return {
    // Raw scores for fine-grained control
    raw: scores,
    
    // Boolean flags for common expressions
    smiling: ((scores.mouthSmileLeft || 0) + (scores.mouthSmileRight || 0)) / 2 > 0.4,
    mouthOpen: (scores.jawOpen || 0) > 0.3,
    leftEyeClosed: (scores.eyeBlinkLeft || 0) > 0.5,
    rightEyeClosed: (scores.eyeBlinkRight || 0) > 0.5,
    bothEyesClosed: (scores.eyeBlinkLeft || 0) > 0.5 && (scores.eyeBlinkRight || 0) > 0.5,
    leftEyebrowRaised: (scores.browOuterUpLeft || 0) > 0.3,
    rightEyebrowRaised: (scores.browOuterUpRight || 0) > 0.3,
    surprised: (scores.browInnerUp || 0) > 0.4 && (scores.jawOpen || 0) > 0.3,
    cheeksPuffed: (scores.cheekPuff || 0) > 0.4,
    mouthPucker: (scores.mouthPucker || 0) > 0.4,
    
    // Numeric values for gradual effects (0-1 range)
    smileAmount: ((scores.mouthSmileLeft || 0) + (scores.mouthSmileRight || 0)) / 2,
    mouthOpenAmount: scores.jawOpen || 0,
    leftEyeOpenAmount: 1 - (scores.eyeBlinkLeft || 0),
    rightEyeOpenAmount: 1 - (scores.eyeBlinkRight || 0),
    browRaiseAmount: (scores.browInnerUp || 0)
  };
}

// Landmark indices for key facial features (same as before for compatibility)
export const FACE_LANDMARKS = {
  // Face oval contour (36 points)
  silhouette: [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
  ],

  // Left eye (16 points)
  leftEye: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
  leftEyeCenter: 468,  // Iris center
  
  // Right eye (16 points)
  rightEye: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
  rightEyeCenter: 473,  // Iris center

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

// List of all 52 blendshape names for reference
export const BLENDSHAPE_NAMES = [
  '_neutral',
  'browDownLeft',
  'browDownRight',
  'browInnerUp',
  'browOuterUpLeft',
  'browOuterUpRight',
  'cheekPuff',
  'cheekSquintLeft',
  'cheekSquintRight',
  'eyeBlinkLeft',
  'eyeBlinkRight',
  'eyeLookDownLeft',
  'eyeLookDownRight',
  'eyeLookInLeft',
  'eyeLookInRight',
  'eyeLookOutLeft',
  'eyeLookOutRight',
  'eyeLookUpLeft',
  'eyeLookUpRight',
  'eyeSquintLeft',
  'eyeSquintRight',
  'eyeWideLeft',
  'eyeWideRight',
  'jawForward',
  'jawLeft',
  'jawOpen',
  'jawRight',
  'mouthClose',
  'mouthDimpleLeft',
  'mouthDimpleRight',
  'mouthFrownLeft',
  'mouthFrownRight',
  'mouthFunnel',
  'mouthLeft',
  'mouthLowerDownLeft',
  'mouthLowerDownRight',
  'mouthPressLeft',
  'mouthPressRight',
  'mouthPucker',
  'mouthRight',
  'mouthRollLower',
  'mouthRollUpper',
  'mouthShrugLower',
  'mouthShrugUpper',
  'mouthSmileLeft',
  'mouthSmileRight',
  'mouthStretchLeft',
  'mouthStretchRight',
  'mouthUpperUpLeft',
  'mouthUpperUpRight',
  'noseSneerLeft',
  'noseSneerRight'
];
