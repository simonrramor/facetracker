// Hand Tracker using MediaPipe Hand Landmarker (tasks-vision)
// Provides 21 hand landmarks per detected hand

// Hand landmark connections for drawing skeleton
export const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring finger
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm
  [5, 9], [9, 13], [13, 17], [0, 17]
];

// Landmark names for reference
export const HAND_LANDMARKS = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20
};

export class HandTracker {
  constructor(videoElement, onResults, onProgress) {
    this.video = videoElement;
    this.onResults = onResults;
    this.onProgress = onProgress;
    this.handLandmarker = null;
    this.running = false;
    this.animationId = null;
    this.lastVideoTime = -1;
  }

  reportProgress(percent, stage) {
    if (this.onProgress) {
      this.onProgress(percent, stage);
    }
  }

  async fetchModelWithProgress(url) {
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch hand model: ${response.status}`);
    }

    const contentLength = response.headers.get('content-length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;
    
    if (!total) {
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
      this.reportProgress(percent, 'hand-model');
    }

    const blob = new Blob(chunks);
    return await blob.arrayBuffer();
  }

  async initialize() {
    console.log('Initializing Hand Landmarker...');
    this.reportProgress(0, 'hand-init');
    
    // Wait for the vision API (already loaded for face tracking)
    const { FilesetResolver } = await window.waitForVisionAPI;
    
    // Also need to import HandLandmarker
    const vision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs');
    const { HandLandmarker } = vision;
    
    try {
      console.log('Loading WASM for hands...');
      this.reportProgress(5, 'hand-wasm');
      
      const wasmFileset = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );
      
      console.log('Downloading hand model...');
      this.reportProgress(15, 'hand-model');

      const modelUrl = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
      
      const modelBuffer = await this.fetchModelWithProgress(modelUrl);
      
      console.log('Hand model downloaded:', modelBuffer.byteLength, 'bytes');
      this.reportProgress(90, 'hand-create');

      this.handLandmarker = await HandLandmarker.createFromOptions(wasmFileset, {
        baseOptions: {
          modelAssetBuffer: new Uint8Array(modelBuffer)
        },
        runningMode: "VIDEO",
        numHands: 2  // Track up to 2 hands
      });

      console.log('Hand Landmarker initialized successfully');
      this.reportProgress(100, 'hand-ready');
      
      return true;
    } catch (error) {
      console.error('Failed to initialize Hand Landmarker:', error);
      throw error;
    }
  }

  detect(timestamp) {
    if (!this.handLandmarker || !this.video || this.video.readyState < 2) {
      return null;
    }

    // Avoid processing the same frame twice
    if (this.video.currentTime === this.lastVideoTime) {
      return null;
    }
    this.lastVideoTime = this.video.currentTime;

    try {
      const results = this.handLandmarker.detectForVideo(this.video, timestamp);
      return this.processResults(results);
    } catch (error) {
      console.error('Hand detection error:', error);
      return null;
    }
  }

  processResults(results) {
    if (!results || !results.landmarks || results.landmarks.length === 0) {
      return {
        hands: [],
        count: 0
      };
    }

    const hands = results.landmarks.map((landmarks, index) => {
      const handedness = results.handednesses?.[index]?.[0];
      
      return {
        landmarks,
        handedness: handedness?.categoryName || 'Unknown',
        confidence: handedness?.score || 0,
        worldLandmarks: results.worldLandmarks?.[index] || null
      };
    });

    return {
      hands,
      count: hands.length
    };
  }

  // Gesture detection helpers
  static isFingerExtended(landmarks, fingerTip, fingerPip) {
    // A finger is extended if tip is higher (lower Y) than PIP joint
    return landmarks[fingerTip].y < landmarks[fingerPip].y;
  }

  static detectGesture(landmarks) {
    if (!landmarks || landmarks.length < 21) return 'unknown';

    const thumbExtended = landmarks[4].x < landmarks[3].x; // Thumb uses X axis
    const indexExtended = this.isFingerExtended(landmarks, 8, 6);
    const middleExtended = this.isFingerExtended(landmarks, 12, 10);
    const ringExtended = this.isFingerExtended(landmarks, 16, 14);
    const pinkyExtended = this.isFingerExtended(landmarks, 20, 18);

    const extendedCount = [indexExtended, middleExtended, ringExtended, pinkyExtended].filter(Boolean).length;

    // Detect common gestures
    if (!thumbExtended && !indexExtended && !middleExtended && !ringExtended && !pinkyExtended) {
      return 'fist';
    }
    if (thumbExtended && indexExtended && middleExtended && ringExtended && pinkyExtended) {
      return 'open';
    }
    if (!thumbExtended && indexExtended && !middleExtended && !ringExtended && !pinkyExtended) {
      return 'pointing';
    }
    if (!thumbExtended && indexExtended && middleExtended && !ringExtended && !pinkyExtended) {
      return 'peace';
    }
    if (thumbExtended && !indexExtended && !middleExtended && !ringExtended && pinkyExtended) {
      return 'rock';
    }
    if (thumbExtended && !indexExtended && !middleExtended && !ringExtended && !pinkyExtended) {
      return 'thumbs_up';
    }
    if (thumbExtended && indexExtended && !middleExtended && !ringExtended && pinkyExtended) {
      return 'love';
    }

    return extendedCount === 0 ? 'fist' : 'partial';
  }

  stop() {
    this.running = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  close() {
    this.stop();
    if (this.handLandmarker) {
      this.handLandmarker.close();
      this.handLandmarker = null;
    }
  }
}
