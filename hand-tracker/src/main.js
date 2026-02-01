// Face Filter App - Main Entry Point
import { FaceTracker, getFaceTransform } from './face-tracker.js';
import { HandTracker } from './hand-tracker.js';
import { MaskRenderer } from './mask-renderer.js';
import { WebGLMaskRenderer } from './webgl-mask-renderer.js';
import { FaceTextureExtractor, loadImageFromFile, loadImageFromURL } from './face-texture-extractor.js';
import { TextureBlender } from './texture-blender.js';
import { ColorMatcher } from './color-matcher.js';
import { moodAnalyzer } from './mood-analyzer.js';

// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const status = document.getElementById('status');
const toggleMeshBtn = document.getElementById('toggleMesh');
const toggleMaskBtn = document.getElementById('toggleMask');
const toggleFaceSwapBtn = document.getElementById('toggleFaceSwap');
const uploadArea = document.getElementById('uploadArea');
const photoUpload = document.getElementById('photoUpload');
const previewContainer = document.getElementById('previewContainer');
const processBtn = document.getElementById('processBtn');
const clearBtn = document.getElementById('clearBtn');
const texturePreview = document.getElementById('texturePreview');
const textureCanvas = document.getElementById('textureCanvas');

// State
let showMesh = true;
let showMask = false;  // Start with mask off
let faceSwapMode = false;
let faceTracker = null;
let handTracker = null;  // Hand tracking
let maskRenderer = null;  // 2D canvas renderer for mesh visualization
let webglMaskRenderer = null;  // WebGL renderer for 3D mask wrapping

// Recording state
let isRecording = false;
let mediaRecorder = null;
let recordedChunks = [];
let recordingCanvas = null;
let recordingCtx = null;

// Custom mesh connection state
let customConnections = [];  // Array of [landmarkIdx1, landmarkIdx2] pairs
let selectedLandmark = null;  // Currently selected landmark index (first click)
let isConnectionMode = false;  // Whether connection mode is active
let currentLandmarks = null;  // Store current frame's landmarks for click detection

// Mesh settings (controllable via UI)
const meshSettings = {
  // Mesh
  lineWidth: 0.5,
  vertexRadius: 1.5,
  meshOpacity: 1,
  showVertices: true,
  showContours: true,
  useDepth: true,
  showTriangles: true,
  strokeColor: '#00ffff',
  fillColor: '#ffffff',
  fillOpacity: 0.05,
  // Adaptive LOD
  useAdaptiveLOD: false,
  enableDenseLandmarks: true,
  enableSubdivision: true,
  lodLevel: 1.0,
  enableSymmetry: true,
  showRegionColors: false,
  // Contours
  contourWidth: 1.5,
  contourColor: '#00ff88',
  showEyes: true,
  showEyebrows: true,
  showLips: true,
  showNose: true,
  showFaceOval: true,
  // Iris tracking - enabled by default
  showIrisTracking: true,
  showGazeDirection: true,
  // Mood detection
  showMoodDetection: true,
  showEmotionWheel: true,
  showEmotionBars: false,
  moodSmoothing: 0.3,
  // Hand tracking
  enableHandTracking: true,
  showHandConnections: true,
  showHandLandmarks: true,
  showHandGestures: true,
  showHandLabels: true,
  // Effects
  showExpressions: true,
  showKeyPoints: true,
  pulseEffect: false,
  animationSpeed: 1,
  // Video
  mirrorVideo: true,
  showVideo: true,
  videoBrightness: 1,
  videoContrast: 1,
  videoSaturation: 1,
  // Display
  showFPS: false,
  showLandmarkIndices: false,
  bgColor: '#000000'
};

// Mask design settings (for procedural masks)
const maskDesign = {
  // Shape
  shape: 'classic',           // classic, angular, masquerade, ninja, phantom
  coverage: 'upper',          // full, upper, lower
  
  // Colors
  primaryColor: '#1a1a2e',    // Main mask color
  secondaryColor: '#16213e',  // Gradient/accent color
  borderColor: '#e94560',     // Border/trim color
  useGradient: true,          // Use gradient fill
  
  // Border
  borderWidth: 3,             // Border thickness
  borderGlow: true,           // Add glow effect to border
  glowIntensity: 0.5,         // Glow strength (0-1)
  
  // Eye holes
  eyeHoleShape: 'pointed',    // oval, pointed, round, narrow
  eyeHoleSize: 1.0,           // Multiplier for eye hole size
  eyeHoleBorder: true,        // Draw border around eye holes
  
  // Surface effects
  surfaceEffect: 'matte',     // matte, metallic, textured
  opacity: 0.95,              // Overall mask opacity
  
  // Decorations
  showPattern: false,         // Add decorative pattern
  patternType: 'none'         // none, dots, lines, scales
};

// Mask shape presets
const maskPresets = {
  classic: {
    shape: 'classic',
    coverage: 'upper',
    primaryColor: '#1a1a2e',
    secondaryColor: '#16213e',
    borderColor: '#e94560',
    useGradient: true,
    borderWidth: 3,
    borderGlow: true,
    glowIntensity: 0.5,
    eyeHoleShape: 'pointed',
    eyeHoleSize: 1.0,
    surfaceEffect: 'matte',
    opacity: 0.95
  },
  angular: {
    shape: 'angular',
    coverage: 'upper',
    primaryColor: '#0f0f0f',
    secondaryColor: '#1a1a1a',
    borderColor: '#ffd700',
    useGradient: true,
    borderWidth: 4,
    borderGlow: true,
    glowIntensity: 0.7,
    eyeHoleShape: 'narrow',
    eyeHoleSize: 0.9,
    surfaceEffect: 'metallic',
    opacity: 1.0
  },
  masquerade: {
    shape: 'masquerade',
    coverage: 'upper',
    primaryColor: '#4a0e4e',
    secondaryColor: '#81007f',
    borderColor: '#ffd700',
    useGradient: true,
    borderWidth: 5,
    borderGlow: true,
    glowIntensity: 0.8,
    eyeHoleShape: 'oval',
    eyeHoleSize: 1.1,
    surfaceEffect: 'metallic',
    opacity: 0.95
  },
  ninja: {
    shape: 'ninja',
    coverage: 'lower',
    primaryColor: '#1a1a1a',
    secondaryColor: '#2d2d2d',
    borderColor: '#333333',
    useGradient: false,
    borderWidth: 0,
    borderGlow: false,
    glowIntensity: 0,
    eyeHoleShape: 'narrow',
    eyeHoleSize: 0,
    surfaceEffect: 'matte',
    opacity: 0.98
  },
  phantom: {
    shape: 'phantom',
    coverage: 'half',
    primaryColor: '#f5f5f5',
    secondaryColor: '#e0e0e0',
    borderColor: '#cccccc',
    useGradient: true,
    borderWidth: 2,
    borderGlow: false,
    glowIntensity: 0,
    eyeHoleShape: 'oval',
    eyeHoleSize: 1.0,
    surfaceEffect: 'matte',
    opacity: 1.0
  }
};

// Accessory/overlay filter settings
const accessorySettings = {
  enabled: false,
  type: 'none',           // none, sunglasses, aviators, hearts, stars, crown, cat_ears, dog_ears, hat
  color: '#000000',       // Tint color for accessories
  opacity: 1.0,
  scale: 1.0
};

// Makeup filter settings
const makeupSettings = {
  enabled: false,
  // Lipstick
  lipstickEnabled: false,
  lipstickColor: '#cc2244',
  lipstickOpacity: 0.7,
  lipstickGloss: true,
  // Eyeliner
  eyelinerEnabled: false,
  eyelinerColor: '#000000',
  eyelinerWidth: 2,
  eyelinerStyle: 'classic',  // classic, wing, smoky
  // Blush
  blushEnabled: false,
  blushColor: '#ff9999',
  blushOpacity: 0.3,
  blushIntensity: 0.5,
  // Eyeshadow
  eyeshadowEnabled: false,
  eyeshadowColor: '#8844aa',
  eyeshadowOpacity: 0.4
};

// Face morphing settings
const morphSettings = {
  enabled: false,
  // Eye size
  eyeSize: 1.0,           // 0.5 = smaller, 2.0 = bigger
  // Nose size
  noseSize: 1.0,          // 0.5 = smaller, 1.5 = bigger
  // Face width
  faceWidth: 1.0,         // 0.8 = narrower, 1.2 = wider
  // Forehead
  foreheadSize: 1.0,
  // Chin
  chinSize: 1.0,
  // Preset effects
  preset: 'none'          // none, cartoon, alien, baby, slim
};

// Morph presets
const morphPresets = {
  none: { eyeSize: 1.0, noseSize: 1.0, faceWidth: 1.0, foreheadSize: 1.0, chinSize: 1.0 },
  cartoon: { eyeSize: 1.6, noseSize: 0.7, faceWidth: 0.9, foreheadSize: 1.1, chinSize: 0.8 },
  alien: { eyeSize: 1.8, noseSize: 0.5, faceWidth: 0.85, foreheadSize: 1.3, chinSize: 0.7 },
  baby: { eyeSize: 1.4, noseSize: 0.8, faceWidth: 1.1, foreheadSize: 1.0, chinSize: 0.9 },
  slim: { eyeSize: 1.0, noseSize: 0.9, faceWidth: 0.85, foreheadSize: 1.0, chinSize: 0.9 }
};

// Settings presets
const presets = {
  wireframe: {
    lineWidth: 0.8,
    vertexRadius: 0,
    meshOpacity: 1,
    showVertices: false,
    showContours: true,
    useDepth: false,
    showTriangles: true,
    strokeColor: '#00ffff',
    fillColor: '#ffffff',
    fillOpacity: 0,
    contourWidth: 1.5,
    contourColor: '#00ffff'
  },
  depth: {
    lineWidth: 0.5,
    vertexRadius: 1.5,
    meshOpacity: 1,
    showVertices: true,
    showContours: true,
    useDepth: true,
    showTriangles: true,
    strokeColor: '#00ffff',
    fillColor: '#ffffff',
    fillOpacity: 0.05,
    contourWidth: 2,
    contourColor: '#00ff88'
  },
  minimal: {
    lineWidth: 0.3,
    vertexRadius: 0,
    meshOpacity: 0.6,
    showVertices: false,
    showContours: true,
    useDepth: false,
    showTriangles: false,
    strokeColor: '#888888',
    fillColor: '#ffffff',
    fillOpacity: 0,
    contourWidth: 1,
    contourColor: '#666666'
  },
  neon: {
    lineWidth: 1.5,
    vertexRadius: 2,
    meshOpacity: 1,
    showVertices: true,
    showContours: true,
    useDepth: false,
    showTriangles: true,
    strokeColor: '#ff00ff',
    fillColor: '#00ff00',
    fillOpacity: 0.1,
    contourWidth: 2.5,
    contourColor: '#ff00ff'
  },
  xray: {
    lineWidth: 0.3,
    vertexRadius: 0.5,
    meshOpacity: 0.8,
    showVertices: true,
    showContours: false,
    useDepth: true,
    showTriangles: true,
    strokeColor: '#00ff00',
    fillColor: '#00ff00',
    fillOpacity: 0.02,
    contourWidth: 1,
    contourColor: '#00ff00',
    showVideo: false,
    bgColor: '#000000'
  },
  retro: {
    lineWidth: 1,
    vertexRadius: 0,
    meshOpacity: 1,
    showVertices: false,
    showContours: true,
    useDepth: false,
    showTriangles: true,
    strokeColor: '#00ff00',
    fillColor: '#003300',
    fillOpacity: 0.15,
    contourWidth: 2,
    contourColor: '#00ff00',
    videoSaturation: 0,
    videoContrast: 1.3
  }
};

// Face swap components
let textureExtractor = null;

// Calculate transform for object-fit: contain (no cropping)
function calculateVideoTransform(videoWidth, videoHeight, displayWidth, displayHeight) {
  return {
    videoWidth, videoHeight,
    displayWidth, displayHeight,
    visibleXStart: 0,
    visibleXRange: 1
  };
}

// === Content Bounds Detection for Virtual Camera Compatibility ===
// Camo Camera and other virtual cameras may add letterboxing (black bars)
// which causes MediaPipe coordinates to not match the displayed video

// Cached content bounds to avoid detecting every frame
let cachedContentBounds = null;
let contentBoundsCacheTime = 0;
const CONTENT_BOUNDS_CACHE_MS = 5000; // Re-detect every 5 seconds

// Check if a pixel is "black" (within threshold for letterbox detection)
function isPixelBlack(data, index, threshold = 30) {
  return data[index] < threshold && 
         data[index + 1] < threshold && 
         data[index + 2] < threshold;
}

// Check if an entire row is mostly black (letterbox bar)
function isRowBlack(data, y, width, threshold = 30, minBlackRatio = 0.9) {
  let blackCount = 0;
  for (let x = 0; x < width; x++) {
    const index = (y * width + x) * 4;
    if (isPixelBlack(data, index, threshold)) blackCount++;
  }
  return blackCount / width >= minBlackRatio;
}

// Check if an entire column is mostly black (letterbox bar)
function isColBlack(data, x, width, height, threshold = 30, minBlackRatio = 0.9) {
  let blackCount = 0;
  for (let y = 0; y < height; y++) {
    const index = (y * width + x) * 4;
    if (isPixelBlack(data, index, threshold)) blackCount++;
  }
  return blackCount / height >= minBlackRatio;
}

// Detect the actual content bounds within the video frame
// Returns { top, bottom, left, right } in pixels
function detectContentBounds(video) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  
  try {
    ctx.drawImage(video, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    const width = canvas.width;
    const height = canvas.height;
    
    // Detect top edge (first non-black row)
    let top = 0;
    for (let y = 0; y < height; y++) {
      if (!isRowBlack(data, y, width)) {
        top = y;
        break;
      }
    }
    
    // Detect bottom edge (last non-black row)
    let bottom = height;
    for (let y = height - 1; y >= 0; y--) {
      if (!isRowBlack(data, y, width)) {
        bottom = y + 1;
        break;
      }
    }
    
    // Detect left edge (first non-black column)
    let left = 0;
    for (let x = 0; x < width; x++) {
      if (!isColBlack(data, x, width, height)) {
        left = x;
        break;
      }
    }
    
    // Detect right edge (last non-black column)
    let right = width;
    for (let x = width - 1; x >= 0; x--) {
      if (!isColBlack(data, x, width, height)) {
        right = x + 1;
        break;
      }
    }
    
    console.log('Detected content bounds:', { top, bottom, left, right, width, height });
    return { top, bottom, left, right, width, height };
  } catch (e) {
    console.warn('Failed to detect content bounds:', e);
    // Return full frame if detection fails
    return { top: 0, bottom: canvas.height, left: 0, right: canvas.width, width: canvas.width, height: canvas.height };
  }
}

// Get cached content bounds, re-detecting periodically
function getContentBounds(video) {
  const now = Date.now();
  if (!cachedContentBounds || (now - contentBoundsCacheTime) > CONTENT_BOUNDS_CACHE_MS) {
    cachedContentBounds = detectContentBounds(video);
    contentBoundsCacheTime = now;
  }
  return cachedContentBounds;
}

// Transform landmarks with dynamic calibration based on video track settings
function transformLandmarksSimple(landmarks, transform, videoElement) {
  // Get actual video track dimensions vs reported dimensions
  let scaleX = 1, scaleY = 1, offsetY = 0;
  
  if (videoElement && videoElement.srcObject) {
    const track = videoElement.srcObject.getVideoTracks()[0];
    if (track) {
      const settings = track.getSettings();
      const reportedWidth = videoElement.videoWidth;
      const reportedHeight = videoElement.videoHeight;
      const actualWidth = settings.width || reportedWidth;
      const actualHeight = settings.height || reportedHeight;
      
      // Calculate scale factors if there's a mismatch
      if (actualWidth !== reportedWidth || actualHeight !== reportedHeight) {
        scaleX = reportedWidth / actualWidth;
        scaleY = reportedHeight / actualHeight;
        console.log('Video dimension mismatch detected:', {
          reported: { w: reportedWidth, h: reportedHeight },
          actual: { w: actualWidth, h: actualHeight },
          scale: { x: scaleX, y: scaleY }
        });
      }
      
      // Check for aspect ratio mismatch (virtual camera cropping)
      const reportedAspect = reportedWidth / reportedHeight;
      const actualAspect = actualWidth / actualHeight;
      if (Math.abs(reportedAspect - actualAspect) > 0.01) {
        // Aspect ratio mismatch - calculate Y offset for top/bottom cropping
        const expectedHeight = reportedWidth / actualAspect;
        offsetY = (expectedHeight - reportedHeight) / (2 * expectedHeight);
        console.log('Aspect ratio mismatch - Y offset:', offsetY);
      }
    }
  }
  
  // Apply X-flip for CSS-mirrored video, and any detected scale/offset corrections
  return landmarks.map(lm => ({
    x: 1 - (lm.x * scaleX),
    y: (lm.y * scaleY) - offsetY,
    z: lm.z
  }));
}

// Set up canvas click handler for custom mesh connections
function setupCanvasClickHandler(canvas, displayWidth, displayHeight) {
  canvas.style.pointerEvents = 'auto';
  canvas.style.cursor = 'crosshair';
  
  canvas.addEventListener('click', (e) => {
    if (!isConnectionMode || !currentLandmarks) return;
    
    // Get click position relative to canvas
    const rect = canvas.getBoundingClientRect();
    const scaleX = displayWidth / rect.width;
    const scaleY = displayHeight / rect.height;
    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;
    
    // Find nearest landmark
    const nearestIdx = findNearestLandmark(clickX, clickY, currentLandmarks, displayWidth, displayHeight);
    
    if (nearestIdx === -1) return;
    
    if (selectedLandmark === null) {
      // First click - select this landmark
      selectedLandmark = nearestIdx;
      status.textContent = `Selected point ${nearestIdx} - click another point to connect`;
    } else {
      // Second click - create connection
      if (nearestIdx !== selectedLandmark) {
        // Check if connection already exists
        const exists = customConnections.some(([a, b]) => 
          (a === selectedLandmark && b === nearestIdx) || 
          (a === nearestIdx && b === selectedLandmark)
        );
        
        if (!exists) {
          customConnections.push([selectedLandmark, nearestIdx]);
          status.textContent = `Connected ${selectedLandmark} to ${nearestIdx} (${customConnections.length} connections)`;
        } else {
          // Remove existing connection
          customConnections = customConnections.filter(([a, b]) => 
            !((a === selectedLandmark && b === nearestIdx) || (a === nearestIdx && b === selectedLandmark))
          );
          status.textContent = `Removed connection (${customConnections.length} connections)`;
        }
      }
      selectedLandmark = null;
    }
  });
}

// Find the nearest landmark to a click position
function findNearestLandmark(clickX, clickY, landmarks, canvasWidth, canvasHeight) {
  let nearestIdx = -1;
  let nearestDist = Infinity;
  const maxDist = 20; // Maximum distance in pixels to consider a click "on" a landmark
  
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    const x = lm.x * canvasWidth;
    const y = lm.y * canvasHeight;
    const dist = Math.sqrt((clickX - x) ** 2 + (clickY - y) ** 2);
    
    if (dist < nearestDist && dist < maxDist) {
      nearestDist = dist;
      nearestIdx = i;
    }
  }
  
  return nearestIdx;
}

// Toggle connection mode
function toggleConnectionMode() {
  isConnectionMode = !isConnectionMode;
  selectedLandmark = null;
  
  const btn = document.getElementById('toggleConnectionMode');
  if (btn) {
    btn.classList.toggle('active', isConnectionMode);
    btn.textContent = isConnectionMode ? 'Exit Connect Mode' : 'Connect Points';
  }
  
  status.textContent = isConnectionMode 
    ? 'Connection mode: Click two points to connect them' 
    : `Connection mode off (${customConnections.length} connections)`;
}

// Clear all custom connections
function clearConnections() {
  customConnections = [];
  selectedLandmark = null;
  status.textContent = 'All custom connections cleared';
}

let textureBlender = null;
let colorMatcher = null;
let uploadedImages = [];  // Array of { file, img, processed: boolean }
let faceSwapTexture = null;  // The final blended face texture
let lastLiveFaceStats = null;  // Color stats from live video

// Initialize settings UI controls
function initSettingsControls() {
  // Slider controls
  const sliders = [
    { id: 'lineWidth', key: 'lineWidth', format: v => v.toFixed(1) },
    { id: 'vertexRadius', key: 'vertexRadius', format: v => v.toFixed(1) },
    { id: 'meshOpacity', key: 'meshOpacity', format: v => Math.round(v * 100) + '%' },
    { id: 'fillOpacity', key: 'fillOpacity', format: v => Math.round(v * 100) + '%' },
    { id: 'lodLevel', key: 'lodLevel', format: v => Math.round(v * 100) + '%' },
    { id: 'contourWidth', key: 'contourWidth', format: v => v.toFixed(1) },
    { id: 'animationSpeed', key: 'animationSpeed', format: v => v.toFixed(1) + 'x' },
    { id: 'moodSmoothing', key: 'moodSmoothing', format: v => Math.round(v * 100) + '%' },
    { id: 'videoBrightness', key: 'videoBrightness', format: v => Math.round(v * 100) + '%' },
    { id: 'videoContrast', key: 'videoContrast', format: v => Math.round(v * 100) + '%' },
    { id: 'videoSaturation', key: 'videoSaturation', format: v => Math.round(v * 100) + '%' }
  ];

  sliders.forEach(({ id, key, format }) => {
    const slider = document.getElementById(id);
    const valueDisplay = document.getElementById(id + 'Value');
    if (slider) {
      slider.value = meshSettings[key];
      if (valueDisplay) valueDisplay.textContent = format(meshSettings[key]);
      
      slider.addEventListener('input', (e) => {
        meshSettings[key] = parseFloat(e.target.value);
        if (valueDisplay) valueDisplay.textContent = format(meshSettings[key]);
        
        // Apply video filters when changed
        if (['videoBrightness', 'videoContrast', 'videoSaturation'].includes(key)) {
          applyVideoFilters();
        }
        
        // Update mood analyzer smoothing
        if (key === 'moodSmoothing') {
          moodAnalyzer.setSmoothingFactor(meshSettings.moodSmoothing);
        }
      });
    }
  });

  // Toggle controls
  const toggles = [
    'showVertices', 'showContours', 'useDepth', 'showTriangles',
    'useAdaptiveLOD', 'enableDenseLandmarks', 'enableSubdivision', 
    'enableSymmetry', 'showRegionColors',
    'showEyes', 'showEyebrows', 'showLips', 'showNose', 'showFaceOval',
    'showIrisTracking', 'showGazeDirection',
    'enableHandTracking', 'showHandConnections', 'showHandLandmarks', 
    'showHandGestures', 'showHandLabels',
    'showMoodDetection', 'showEmotionWheel', 'showEmotionBars',
    'showExpressions', 'showKeyPoints', 'pulseEffect',
    'mirrorVideo', 'showVideo', 'showFPS', 'showLandmarkIndices'
  ];
  
  toggles.forEach(key => {
    const toggle = document.getElementById(key);
    if (toggle) {
      toggle.checked = meshSettings[key];
      toggle.addEventListener('change', async (e) => {
        meshSettings[key] = e.target.checked;
        
        // Apply special effects
        if (key === 'mirrorVideo') {
          video.style.transform = e.target.checked ? 'scaleX(-1)' : 'none';
        }
        if (key === 'showVideo') {
          video.style.opacity = e.target.checked ? '1' : '0';
        }
        
        // Initialize hand tracking when enabled
        if (key === 'enableHandTracking' && e.target.checked) {
          await initializeHandTracking();
        }
        
      });
    }
  });

  // Color pickers
  const colors = [
    { id: 'strokeColor', key: 'strokeColor' },
    { id: 'fillColor', key: 'fillColor' },
    { id: 'contourColor', key: 'contourColor' },
    { id: 'bgColor', key: 'bgColor' }
  ];

  colors.forEach(({ id, key }) => {
    const picker = document.getElementById(id);
    const valueDisplay = document.getElementById(id + 'Value');
    if (picker) {
      picker.value = meshSettings[key];
      if (valueDisplay) valueDisplay.textContent = meshSettings[key];
      
      picker.addEventListener('input', (e) => {
        meshSettings[key] = e.target.value;
        if (valueDisplay) valueDisplay.textContent = e.target.value;
        
        // Apply background color
        if (key === 'bgColor') {
          document.body.style.background = e.target.value;
        }
      });
    }
  });

  // Preset buttons
  const presetButtons = {
    'presetWireframe': 'wireframe',
    'presetDepth': 'depth',
    'presetMinimal': 'minimal',
    'presetNeon': 'neon',
    'presetXray': 'xray',
    'presetRetro': 'retro'
  };

  Object.entries(presetButtons).forEach(([btnId, presetName]) => {
    const btn = document.getElementById(btnId);
    if (btn) {
      btn.addEventListener('click', () => {
        applyPreset(presetName);
        // Update active state
        document.querySelectorAll('.preset-btn:not(.mask-preset-btn)').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    }
  });
  
  // Initialize mask designer controls
  initMaskDesignerControls();
}

// Initialize mask designer UI controls
function initMaskDesignerControls() {
  // Mask preset buttons
  const maskPresetButtons = {
    'maskClassic': 'classic',
    'maskAngular': 'angular',
    'maskMasquerade': 'masquerade',
    'maskNinja': 'ninja',
    'maskPhantom': 'phantom'
  };

  Object.entries(maskPresetButtons).forEach(([btnId, presetName]) => {
    const btn = document.getElementById(btnId);
    if (btn) {
      btn.addEventListener('click', () => {
        applyMaskPreset(presetName);
        // Update active state
        document.querySelectorAll('.mask-preset-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    }
  });

  // Mask sliders
  const maskSliders = [
    { id: 'maskBorderWidth', key: 'borderWidth', format: v => v.toFixed(1) },
    { id: 'maskGlowIntensity', key: 'glowIntensity', format: v => Math.round(v * 100) + '%' },
    { id: 'maskEyeHoleSize', key: 'eyeHoleSize', format: v => v.toFixed(1) + 'x' },
    { id: 'maskOpacity', key: 'opacity', format: v => Math.round(v * 100) + '%' }
  ];

  maskSliders.forEach(({ id, key, format }) => {
    const slider = document.getElementById(id);
    const valueDisplay = document.getElementById(id + 'Value');
    if (slider) {
      slider.value = maskDesign[key];
      if (valueDisplay) valueDisplay.textContent = format(maskDesign[key]);
      
      slider.addEventListener('input', (e) => {
        maskDesign[key] = parseFloat(e.target.value);
        if (valueDisplay) valueDisplay.textContent = format(maskDesign[key]);
      });
    }
  });

  // Mask toggles
  const maskToggles = ['maskUseGradient', 'maskBorderGlow', 'maskEyeHoleBorder'];
  const maskToggleKeys = { 
    'maskUseGradient': 'useGradient', 
    'maskBorderGlow': 'borderGlow', 
    'maskEyeHoleBorder': 'eyeHoleBorder' 
  };
  
  maskToggles.forEach(id => {
    const toggle = document.getElementById(id);
    const key = maskToggleKeys[id];
    if (toggle && key) {
      toggle.checked = maskDesign[key];
      toggle.addEventListener('change', (e) => {
        maskDesign[key] = e.target.checked;
      });
    }
  });

  // Mask color pickers
  const maskColors = [
    { id: 'maskPrimaryColor', key: 'primaryColor' },
    { id: 'maskSecondaryColor', key: 'secondaryColor' },
    { id: 'maskBorderColor', key: 'borderColor' }
  ];

  maskColors.forEach(({ id, key }) => {
    const picker = document.getElementById(id);
    const valueDisplay = document.getElementById(id + 'Value');
    if (picker) {
      picker.value = maskDesign[key];
      if (valueDisplay) valueDisplay.textContent = maskDesign[key];
      
      picker.addEventListener('input', (e) => {
        maskDesign[key] = e.target.value;
        if (valueDisplay) valueDisplay.textContent = e.target.value;
      });
    }
  });
}

// Apply a mask preset
function applyMaskPreset(presetName) {
  const preset = maskPresets[presetName];
  if (!preset) return;

  // Update maskDesign with preset values
  Object.assign(maskDesign, preset);

  // Update UI elements
  const sliderUpdates = [
    { id: 'maskBorderWidth', key: 'borderWidth', format: v => v.toFixed(1) },
    { id: 'maskGlowIntensity', key: 'glowIntensity', format: v => Math.round(v * 100) + '%' },
    { id: 'maskEyeHoleSize', key: 'eyeHoleSize', format: v => v.toFixed(1) + 'x' },
    { id: 'maskOpacity', key: 'opacity', format: v => Math.round(v * 100) + '%' }
  ];

  sliderUpdates.forEach(({ id, key, format }) => {
    const slider = document.getElementById(id);
    const valueDisplay = document.getElementById(id + 'Value');
    if (slider && maskDesign[key] !== undefined) {
      slider.value = maskDesign[key];
      if (valueDisplay) valueDisplay.textContent = format(maskDesign[key]);
    }
  });

  // Update toggles
  const maskToggles = { 
    'maskUseGradient': 'useGradient', 
    'maskBorderGlow': 'borderGlow', 
    'maskEyeHoleBorder': 'eyeHoleBorder' 
  };
  
  Object.entries(maskToggles).forEach(([id, key]) => {
    const toggle = document.getElementById(id);
    if (toggle && maskDesign[key] !== undefined) {
      toggle.checked = maskDesign[key];
    }
  });

  // Update color pickers
  const colorUpdates = [
    { id: 'maskPrimaryColor', key: 'primaryColor' },
    { id: 'maskSecondaryColor', key: 'secondaryColor' },
    { id: 'maskBorderColor', key: 'borderColor' }
  ];

  colorUpdates.forEach(({ id, key }) => {
    const picker = document.getElementById(id);
    const valueDisplay = document.getElementById(id + 'Value');
    if (picker && maskDesign[key]) {
      picker.value = maskDesign[key];
      if (valueDisplay) valueDisplay.textContent = maskDesign[key];
    }
  });
}

// Initialize accessory controls
function initAccessoryControls() {
  // Accessory type dropdown
  const accessoryType = document.getElementById('accessoryType');
  if (accessoryType) {
    accessoryType.value = accessorySettings.type;
    accessoryType.addEventListener('change', (e) => {
      accessorySettings.type = e.target.value;
      accessorySettings.enabled = e.target.value !== 'none';
    });
  }

  // Accessory color
  const accessoryColor = document.getElementById('accessoryColor');
  const accessoryColorValue = document.getElementById('accessoryColorValue');
  if (accessoryColor) {
    accessoryColor.value = accessorySettings.color;
    if (accessoryColorValue) accessoryColorValue.textContent = accessorySettings.color;
    accessoryColor.addEventListener('input', (e) => {
      accessorySettings.color = e.target.value;
      if (accessoryColorValue) accessoryColorValue.textContent = e.target.value;
    });
  }

  // Accessory scale
  const accessoryScale = document.getElementById('accessoryScale');
  const accessoryScaleValue = document.getElementById('accessoryScaleValue');
  if (accessoryScale) {
    accessoryScale.value = accessorySettings.scale;
    if (accessoryScaleValue) accessoryScaleValue.textContent = accessorySettings.scale.toFixed(1) + 'x';
    accessoryScale.addEventListener('input', (e) => {
      accessorySettings.scale = parseFloat(e.target.value);
      if (accessoryScaleValue) accessoryScaleValue.textContent = accessorySettings.scale.toFixed(1) + 'x';
    });
  }
}

// Initialize makeup controls
function initMakeupControls() {
  // Lipstick toggle
  const lipstickEnabled = document.getElementById('lipstickEnabled');
  if (lipstickEnabled) {
    lipstickEnabled.checked = makeupSettings.lipstickEnabled;
    lipstickEnabled.addEventListener('change', (e) => {
      makeupSettings.lipstickEnabled = e.target.checked;
      makeupSettings.enabled = makeupSettings.lipstickEnabled || makeupSettings.eyelinerEnabled || 
                               makeupSettings.blushEnabled || makeupSettings.eyeshadowEnabled;
    });
  }

  // Lipstick color
  const lipstickColor = document.getElementById('lipstickColor');
  const lipstickColorValue = document.getElementById('lipstickColorValue');
  if (lipstickColor) {
    lipstickColor.value = makeupSettings.lipstickColor;
    if (lipstickColorValue) lipstickColorValue.textContent = makeupSettings.lipstickColor;
    lipstickColor.addEventListener('input', (e) => {
      makeupSettings.lipstickColor = e.target.value;
      if (lipstickColorValue) lipstickColorValue.textContent = e.target.value;
    });
  }

  // Eyeliner toggle
  const eyelinerEnabled = document.getElementById('eyelinerEnabled');
  if (eyelinerEnabled) {
    eyelinerEnabled.checked = makeupSettings.eyelinerEnabled;
    eyelinerEnabled.addEventListener('change', (e) => {
      makeupSettings.eyelinerEnabled = e.target.checked;
      makeupSettings.enabled = makeupSettings.lipstickEnabled || makeupSettings.eyelinerEnabled || 
                               makeupSettings.blushEnabled || makeupSettings.eyeshadowEnabled;
    });
  }

  // Eyeliner style
  const eyelinerStyle = document.getElementById('eyelinerStyle');
  if (eyelinerStyle) {
    eyelinerStyle.value = makeupSettings.eyelinerStyle;
    eyelinerStyle.addEventListener('change', (e) => {
      makeupSettings.eyelinerStyle = e.target.value;
    });
  }

  // Blush toggle
  const blushEnabled = document.getElementById('blushEnabled');
  if (blushEnabled) {
    blushEnabled.checked = makeupSettings.blushEnabled;
    blushEnabled.addEventListener('change', (e) => {
      makeupSettings.blushEnabled = e.target.checked;
      makeupSettings.enabled = makeupSettings.lipstickEnabled || makeupSettings.eyelinerEnabled || 
                               makeupSettings.blushEnabled || makeupSettings.eyeshadowEnabled;
    });
  }

  // Blush color
  const blushColor = document.getElementById('blushColor');
  const blushColorValue = document.getElementById('blushColorValue');
  if (blushColor) {
    blushColor.value = makeupSettings.blushColor;
    if (blushColorValue) blushColorValue.textContent = makeupSettings.blushColor;
    blushColor.addEventListener('input', (e) => {
      makeupSettings.blushColor = e.target.value;
      if (blushColorValue) blushColorValue.textContent = e.target.value;
    });
  }

  // Eyeshadow toggle
  const eyeshadowEnabled = document.getElementById('eyeshadowEnabled');
  if (eyeshadowEnabled) {
    eyeshadowEnabled.checked = makeupSettings.eyeshadowEnabled;
    eyeshadowEnabled.addEventListener('change', (e) => {
      makeupSettings.eyeshadowEnabled = e.target.checked;
      makeupSettings.enabled = makeupSettings.lipstickEnabled || makeupSettings.eyelinerEnabled || 
                               makeupSettings.blushEnabled || makeupSettings.eyeshadowEnabled;
    });
  }

  // Eyeshadow color
  const eyeshadowColor = document.getElementById('eyeshadowColor');
  const eyeshadowColorValue = document.getElementById('eyeshadowColorValue');
  if (eyeshadowColor) {
    eyeshadowColor.value = makeupSettings.eyeshadowColor;
    if (eyeshadowColorValue) eyeshadowColorValue.textContent = makeupSettings.eyeshadowColor;
    eyeshadowColor.addEventListener('input', (e) => {
      makeupSettings.eyeshadowColor = e.target.value;
      if (eyeshadowColorValue) eyeshadowColorValue.textContent = e.target.value;
    });
  }
}

// Initialize morph controls
function initMorphControls() {
  // Morph preset buttons
  const morphPresetButtons = {
    'morphNone': 'none',
    'morphCartoon': 'cartoon',
    'morphAlien': 'alien',
    'morphBaby': 'baby',
    'morphSlim': 'slim'
  };

  Object.entries(morphPresetButtons).forEach(([btnId, presetName]) => {
    const btn = document.getElementById(btnId);
    if (btn) {
      btn.addEventListener('click', () => {
        applyMorphPreset(presetName);
        document.querySelectorAll('.morph-preset-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    }
  });

  // Morph sliders
  const morphSliders = [
    { id: 'morphEyeSize', key: 'eyeSize' },
    { id: 'morphNoseSize', key: 'noseSize' },
    { id: 'morphFaceWidth', key: 'faceWidth' },
    { id: 'morphChinSize', key: 'chinSize' }
  ];

  morphSliders.forEach(({ id, key }) => {
    const slider = document.getElementById(id);
    const valueDisplay = document.getElementById(id + 'Value');
    if (slider) {
      slider.value = morphSettings[key];
      if (valueDisplay) valueDisplay.textContent = morphSettings[key].toFixed(1) + 'x';
      
      slider.addEventListener('input', (e) => {
        morphSettings[key] = parseFloat(e.target.value);
        morphSettings.enabled = morphSettings.eyeSize !== 1.0 || morphSettings.noseSize !== 1.0 ||
                                morphSettings.faceWidth !== 1.0 || morphSettings.chinSize !== 1.0;
        if (valueDisplay) valueDisplay.textContent = morphSettings[key].toFixed(1) + 'x';
      });
    }
  });
}

// Apply morph preset
function applyMorphPreset(presetName) {
  const preset = morphPresets[presetName];
  if (!preset) return;

  Object.assign(morphSettings, preset);
  morphSettings.enabled = presetName !== 'none';
  morphSettings.preset = presetName;

  // Update sliders
  const morphSliders = [
    { id: 'morphEyeSize', key: 'eyeSize' },
    { id: 'morphNoseSize', key: 'noseSize' },
    { id: 'morphFaceWidth', key: 'faceWidth' },
    { id: 'morphChinSize', key: 'chinSize' }
  ];

  morphSliders.forEach(({ id, key }) => {
    const slider = document.getElementById(id);
    const valueDisplay = document.getElementById(id + 'Value');
    if (slider && morphSettings[key] !== undefined) {
      slider.value = morphSettings[key];
      if (valueDisplay) valueDisplay.textContent = morphSettings[key].toFixed(1) + 'x';
    }
  });
}

// Apply video CSS filters
function applyVideoFilters() {
  const { videoBrightness, videoContrast, videoSaturation } = meshSettings;
  video.style.filter = `brightness(${videoBrightness}) contrast(${videoContrast}) saturate(${videoSaturation})`;
}


// Initialize recording canvas (composites video + mesh overlay)
function initRecordingCanvas() {
  recordingCanvas = document.createElement('canvas');
  recordingCanvas.width = 378;
  recordingCanvas.height = 756;
  recordingCtx = recordingCanvas.getContext('2d');
}

// Update recording canvas each frame
function updateRecordingCanvas() {
  if (!recordingCtx || !isRecording) return;
  
  // Clear
  recordingCtx.clearRect(0, 0, recordingCanvas.width, recordingCanvas.height);
  
  // Draw video (mirrored if needed)
  recordingCtx.save();
  if (meshSettings.mirrorVideo) {
    recordingCtx.scale(-1, 1);
    recordingCtx.translate(-recordingCanvas.width, 0);
  }
  
  // Apply video filters
  recordingCtx.filter = `brightness(${meshSettings.videoBrightness}) contrast(${meshSettings.videoContrast}) saturate(${meshSettings.videoSaturation})`;
  
  // Calculate crop for object-fit: cover
  const videoAspect = video.videoWidth / video.videoHeight;
  const canvasAspect = recordingCanvas.width / recordingCanvas.height;
  
  let sx, sy, sw, sh;
  if (videoAspect > canvasAspect) {
    // Video is wider - crop sides
    sh = video.videoHeight;
    sw = sh * canvasAspect;
    sx = (video.videoWidth - sw) / 2;
    sy = 0;
  } else {
    // Video is taller - crop top/bottom
    sw = video.videoWidth;
    sh = sw / canvasAspect;
    sx = 0;
    sy = (video.videoHeight - sh) / 2;
  }
  
  if (meshSettings.showVideo) {
    recordingCtx.drawImage(video, sx, sy, sw, sh, 0, 0, recordingCanvas.width, recordingCanvas.height);
  }
  
  recordingCtx.filter = 'none';
  recordingCtx.restore();
  
  // Draw mesh overlay (already correctly positioned)
  recordingCtx.drawImage(canvas, 0, 0);
}

// Start recording
function startRecording() {
  if (isRecording) return;
  
  initRecordingCanvas();
  recordedChunks = [];
  
  // Get stream from recording canvas
  const stream = recordingCanvas.captureStream(30); // 30 FPS
  
  // Try different codecs
  const mimeTypes = [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
    'video/mp4'
  ];
  
  let selectedMime = mimeTypes.find(mime => MediaRecorder.isTypeSupported(mime)) || 'video/webm';
  
  mediaRecorder = new MediaRecorder(stream, { 
    mimeType: selectedMime,
    videoBitsPerSecond: 5000000 // 5 Mbps
  });
  
  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) {
      recordedChunks.push(e.data);
    }
  };
  
  mediaRecorder.onstop = () => {
    // Create blob and download
    const blob = new Blob(recordedChunks, { type: selectedMime });
    const url = URL.createObjectURL(blob);
    
    // Create download link
    const a = document.createElement('a');
    a.href = url;
    a.download = `face-recording-${Date.now()}.webm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    // Cleanup
    URL.revokeObjectURL(url);
    recordedChunks = [];
  };
  
  mediaRecorder.start(100); // Collect data every 100ms
  isRecording = true;
  
  // Update button
  const recordBtn = document.getElementById('recordBtn');
  if (recordBtn) {
    recordBtn.textContent = 'Stop';
    recordBtn.classList.add('recording');
  }
  
  console.log('Recording started with', selectedMime);
}

// Stop recording
function stopRecording() {
  if (!isRecording || !mediaRecorder) return;
  
  mediaRecorder.stop();
  isRecording = false;
  
  // Update button
  const recordBtn = document.getElementById('recordBtn');
  if (recordBtn) {
    recordBtn.textContent = 'Record';
    recordBtn.classList.remove('recording');
  }
  
  console.log('Recording stopped');
}

// Toggle recording
function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}

// Apply a preset to settings and update UI
function applyPreset(presetName) {
  const preset = presets[presetName];
  if (!preset) return;

  // Update settings
  Object.assign(meshSettings, preset);

  // Update slider UIs
  const sliderUpdates = [
    { id: 'lineWidth', format: v => v.toFixed(1) },
    { id: 'vertexRadius', format: v => v.toFixed(1) },
    { id: 'meshOpacity', format: v => Math.round(v * 100) + '%' },
    { id: 'fillOpacity', format: v => Math.round(v * 100) + '%' },
    { id: 'lodLevel', format: v => Math.round(v * 100) + '%' },
    { id: 'contourWidth', format: v => v.toFixed(1) },
    { id: 'animationSpeed', format: v => v.toFixed(1) + 'x' },
    { id: 'moodSmoothing', format: v => Math.round(v * 100) + '%' },
    { id: 'videoBrightness', format: v => Math.round(v * 100) + '%' },
    { id: 'videoContrast', format: v => Math.round(v * 100) + '%' },
    { id: 'videoSaturation', format: v => Math.round(v * 100) + '%' }
  ];

  sliderUpdates.forEach(({ id, format }) => {
    const slider = document.getElementById(id);
    const valueDisplay = document.getElementById(id + 'Value');
    if (slider && meshSettings[id] !== undefined) {
      slider.value = meshSettings[id];
      if (valueDisplay) valueDisplay.textContent = format(meshSettings[id]);
    }
  });

  // Update toggle UIs
  const allToggles = [
    'showVertices', 'showContours', 'useDepth', 'showTriangles',
    'useAdaptiveLOD', 'enableDenseLandmarks', 'enableSubdivision',
    'enableSymmetry', 'showRegionColors',
    'showEyes', 'showEyebrows', 'showLips', 'showNose', 'showFaceOval',
    'showIrisTracking', 'showGazeDirection',
    'enableHandTracking', 'showHandConnections', 'showHandLandmarks',
    'showHandGestures', 'showHandLabels',
    'showMoodDetection', 'showEmotionWheel', 'showEmotionBars',
    'showExpressions', 'showKeyPoints', 'pulseEffect',
    'mirrorVideo', 'showVideo', 'showFPS', 'showLandmarkIndices'
  ];
  
  allToggles.forEach(key => {
    const toggle = document.getElementById(key);
    if (toggle && meshSettings[key] !== undefined) {
      toggle.checked = meshSettings[key];
    }
  });

  // Update color picker UIs
  ['strokeColor', 'fillColor', 'contourColor', 'bgColor'].forEach(key => {
    const picker = document.getElementById(key);
    const valueDisplay = document.getElementById(key + 'Value');
    if (picker && meshSettings[key]) {
      picker.value = meshSettings[key];
      if (valueDisplay) valueDisplay.textContent = meshSettings[key];
    }
  });
  
  // Apply video effects
  applyVideoFilters();
  video.style.opacity = meshSettings.showVideo ? '1' : '0';
  video.style.transform = meshSettings.mirrorVideo ? 'scaleX(-1)' : 'none';
  document.body.style.background = meshSettings.bgColor;
}

// Initialize hand tracking (on demand)
async function initializeHandTracking() {
  if (handTracker) {
    console.log('Hand tracker already initialized');
    return;
  }
  
  console.log('Initializing hand tracking...');
  status.textContent = 'Loading hand tracking model...';
  
  try {
    handTracker = new HandTracker(video, null, (percent, stage) => {
      status.textContent = `Loading hands: ${percent}%`;
    });
    
    await handTracker.initialize();
    status.textContent = 'Hand tracking ready!';
    console.log('Hand tracking initialized successfully');
    
    setTimeout(() => {
      status.textContent = 'Tracking...';
    }, 1500);
  } catch (error) {
    console.error('Failed to initialize hand tracking:', error);
    status.textContent = 'Hand tracking failed to load';
    meshSettings.enableHandTracking = false;
    const toggle = document.getElementById('enableHandTracking');
    if (toggle) toggle.checked = false;
  }
}

// Initialize the app
async function init() {
  console.log('Starting initialization...');
  status.textContent = 'Requesting camera...';
  
  // Request camera FIRST - this is the most important step
  try {
    console.log('Calling getUserMedia...');
    // Request high quality from webcam
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { 
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user' 
      }
    });
    console.log('Got camera stream:', stream);
    video.srcObject = stream;
    await video.play();
    console.log('Camera playing');
    console.log('Video actual size:', video.videoWidth, 'x', video.videoHeight);
    status.textContent = 'Camera active!';
  } catch (camError) {
    console.error('Camera error:', camError);
    status.textContent = 'Camera error: ' + camError.message;
    status.style.color = '#ff6b6b';
    return;
  }

  try {
    // Get actual video dimensions from the camera stream
    const actualVideoWidth = video.videoWidth;
    const actualVideoHeight = video.videoHeight;
    console.log('Video actual size:', actualVideoWidth, 'x', actualVideoHeight);
    
    // Calculate display size for the container
    const maxWidth = 960;
    const maxHeight = 700;
    const videoAspect = actualVideoWidth / actualVideoHeight;
    
    let displayWidth, displayHeight;
    if (actualVideoWidth / maxWidth > actualVideoHeight / maxHeight) {
      displayWidth = maxWidth;
      displayHeight = Math.round(maxWidth / videoAspect);
    } else {
      displayHeight = maxHeight;
      displayWidth = Math.round(maxHeight * videoAspect);
    }
    
    // Set canvas to match video dimensions for 1:1 coordinate mapping
    canvas.width = actualVideoWidth;
    canvas.height = actualVideoHeight;
    
    // Store scale factors for coordinate transformation
    window.videoScaleX = displayWidth / actualVideoWidth;
    window.videoScaleY = displayHeight / actualVideoHeight;
    
    // Update the video-wrap container size
    const videoWrap = document.querySelector('.video-wrap');
    if (videoWrap) {
      videoWrap.style.width = displayWidth + 'px';
      videoWrap.style.height = displayHeight + 'px';
    }
    
    console.log('Video actual:', actualVideoWidth, 'x', actualVideoHeight);
    console.log('Canvas/Display:', canvas.width, 'x', canvas.height);
    console.log('Scale factors:', window.videoScaleX, window.videoScaleY);
    
    status.textContent = `Video: ${actualVideoWidth}x${actualVideoHeight} | Canvas: ${displayWidth}x${displayHeight}`;
    
    // No transform needed - canvas matches video exactly
    window.videoTransform = calculateVideoTransform(
      actualVideoWidth, actualVideoHeight,
      actualVideoWidth, actualVideoHeight  // Same dimensions = no transform
    );
    console.log('Video transform:', window.videoTransform);

    // Create 2D mask renderer for mesh visualization
    maskRenderer = new MaskRenderer(canvas);
    console.log('2D mask renderer created');
    
    // Set up click handler for custom mesh connections
    setupCanvasClickHandler(canvas, actualVideoWidth, actualVideoHeight);

    // Create WebGL canvas for 3D mask rendering
    status.textContent = 'Setting up WebGL...';
    console.log('Video wrap element:', videoWrap);
    console.log('Video wrap children before:', videoWrap.children.length);
    
    // Check if there's already a webgl-canvas and remove it
    const existingWebgl = document.getElementById('webgl-canvas');
    if (existingWebgl) {
      console.log('Removing existing WebGL canvas');
      existingWebgl.remove();
    }
    
    const webglCanvas = document.createElement('canvas');
    webglCanvas.id = 'webgl-canvas';
    webglCanvas.width = actualVideoWidth;
    webglCanvas.height = actualVideoHeight;
    // Use same styling as the 2D canvas
    webglCanvas.style.position = 'absolute';
    webglCanvas.style.top = '0';
    webglCanvas.style.left = '0';
    // No CSS mirroring - handled in JavaScript
    webglCanvas.style.pointerEvents = 'none';
    webglCanvas.style.zIndex = '20';
    
    // Insert after the 2D canvas
    videoWrap.appendChild(webglCanvas);
    console.log('WebGL canvas appended, parent:', webglCanvas.parentElement?.className);
    console.log('Video wrap children after:', videoWrap.children.length);
    console.log('WebGL canvas in DOM:', document.getElementById('webgl-canvas') !== null);
    
    // Create WebGL mask renderer
    console.log('Creating WebGL renderer...');
    try {
      webglMaskRenderer = new WebGLMaskRenderer(webglCanvas);
      console.log('WebGL renderer created successfully');

      // Generate procedural mask (skip trying to load files)
      console.log('Generating procedural mask...');
      webglMaskRenderer.loadProceduralMask();
      console.log('Procedural mask loaded, maskLoaded =', webglMaskRenderer.maskLoaded);
    } catch (webglError) {
      console.error('WebGL error:', webglError);
      webglMaskRenderer = null;
    }

    // Create face tracker with progress callback
    status.textContent = 'Loading Face Landmarker...';
    console.log('Creating face tracker...');
    
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    
    // Progress callback to update UI
    const onProgress = (percent, stage) => {
      if (progressContainer) progressContainer.style.display = 'block';
      if (progressBar) progressBar.style.width = percent + '%';
      
      const stageNames = {
        'init': 'Initializing...',
        'wasm': 'Loading WASM runtime...',
        'model': `Downloading model... ${percent}%`,
        'create': 'Creating detector...',
        'done': 'Ready!'
      };
      status.textContent = stageNames[stage] || `Loading... ${percent}%`;
    };
    
    faceTracker = new FaceTracker(video, onFaceResults, onProgress);

    try {
      await faceTracker.initialize();
      console.log('Face tracker initialized');
      
      // Hide progress bar and show ready message
      if (progressContainer) progressContainer.style.display = 'none';
      status.textContent = 'Ready - 478 landmarks + 52 expressions';
      status.style.color = '#4CAF50';
      
      // Initialize hand tracking if enabled by default
      if (meshSettings.enableHandTracking) {
        await initializeHandTracking();
      }
    } catch (initError) {
      if (progressContainer) progressContainer.style.display = 'none';
      console.error('Face tracker init failed:', initError);
      status.textContent = 'Error: ' + initError.message;
      status.style.color = '#ff6b6b';
      return;
    }

    // Set up toggle buttons
    if (toggleMeshBtn) {
      toggleMeshBtn.addEventListener('click', () => {
        showMesh = !showMesh;
        toggleMeshBtn.classList.toggle('active', showMesh);
      });
    }

    if (toggleMaskBtn) {
      toggleMaskBtn.addEventListener('click', () => {
        showMask = !showMask;
        toggleMaskBtn.classList.toggle('active', showMask);
      });
    }

    // Face swap toggle
    if (toggleFaceSwapBtn) {
      toggleFaceSwapBtn.addEventListener('click', () => {
        faceSwapMode = !faceSwapMode;
        toggleFaceSwapBtn.classList.toggle('active', faceSwapMode);
        if (faceSwapMode && !faceSwapTexture) {
          status.textContent = 'Upload and process photos first!';
        }
      });
    }

    // Record button
    const recordBtn = document.getElementById('recordBtn');
    if (recordBtn) {
      recordBtn.addEventListener('click', toggleRecording);
    }

    // Connection mode buttons
    const toggleConnectionModeBtn = document.getElementById('toggleConnectionMode');
    if (toggleConnectionModeBtn) {
      toggleConnectionModeBtn.addEventListener('click', toggleConnectionMode);
    }
    
    const clearConnectionsBtn = document.getElementById('clearConnections');
    if (clearConnectionsBtn) {
      clearConnectionsBtn.addEventListener('click', clearConnections);
    }

    // Initialize settings controls
    initSettingsControls();
    
    
    // Initialize new filter controls
    initAccessoryControls();
    initMakeupControls();
    initMorphControls();

    // Initialize face swap components
    console.log('Creating face swap components...');
    textureExtractor = new FaceTextureExtractor();
    textureBlender = new TextureBlender(512);
    colorMatcher = new ColorMatcher();
    console.log('Face swap components created');

    // Set up upload handlers
    console.log('Setting up upload handlers...');
    setupUploadHandlers();

  } catch (error) {
    console.error('Initialization error:', error);
    status.textContent = 'Error: ' + error.message;
    status.style.color = '#ff6b6b';
  }
}

// Set up photo upload handlers
function setupUploadHandlers() {
  // Click to upload
  if (uploadArea) {
    uploadArea.addEventListener('click', () => {
      photoUpload?.click();
    });
  }

  // File input change
  if (photoUpload) {
    photoUpload.addEventListener('change', async (e) => {
      const files = Array.from(e.target.files);
      for (const file of files) {
        await addUploadedImage(file);
      }
    });
  }

  // Drag and drop
  if (uploadArea) {
    uploadArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadArea.style.borderColor = '#00ff88';
    });

    uploadArea.addEventListener('dragleave', () => {
      uploadArea.style.borderColor = '';
    });

    uploadArea.addEventListener('drop', async (e) => {
      e.preventDefault();
      uploadArea.style.borderColor = '';
      const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
      for (const file of files) {
        await addUploadedImage(file);
      }
    });
  }

  // Process button
  if (processBtn) {
    console.log('Process button found, attaching event listener');
    processBtn.addEventListener('click', () => {
      console.log('Process button clicked!');
      processUploadedImages();
    });
  } else {
    console.error('Process button not found!');
  }

  // Clear button
  if (clearBtn) {
    clearBtn.addEventListener('click', clearUploadedImages);
  } else {
    console.error('Clear button not found!');
  }
  
  console.log('Upload handlers setup complete');
}

// Add an uploaded image to the list
async function addUploadedImage(file) {
  try {
    const img = await loadImageFromFile(file);
    const entry = { file, img, processed: false };
    uploadedImages.push(entry);
    
    // Create preview
    const previewItem = document.createElement('div');
    previewItem.className = 'preview-item';
    previewItem.innerHTML = `
      <img src="${img.src}" alt="Face photo">
      <button class="remove-btn">×</button>
    `;
    
    previewItem.querySelector('.remove-btn').addEventListener('click', () => {
      const idx = uploadedImages.indexOf(entry);
      if (idx >= 0) {
        uploadedImages.splice(idx, 1);
        previewItem.remove();
      }
    });
    
    previewContainer?.appendChild(previewItem);
    entry.previewElement = previewItem;
    
    console.log('Added image:', file.name, img.width, 'x', img.height);
  } catch (err) {
    console.error('Failed to load image:', err);
    status.textContent = 'Failed to load image';
  }
}

// Python backend URL for face extraction
const FACE_EXTRACTOR_API = 'http://localhost:5000';

// Check if Python backend is available
async function checkPythonBackend() {
  try {
    const response = await fetch(`${FACE_EXTRACTOR_API}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

// Extract face using Python backend
async function extractFaceWithBackend(imageDataUrl) {
  const response = await fetch(`${FACE_EXTRACTOR_API}/extract`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageDataUrl, padding: 1.3 })
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Face extraction failed');
  }
  
  return response.json();
}

// Process all uploaded images
async function processUploadedImages() {
  console.log('processUploadedImages called, images:', uploadedImages.length);
  
  if (uploadedImages.length === 0) {
    status.textContent = 'No images to process';
    return;
  }

  // Check if Python backend is available
  const useBackend = await checkPythonBackend();
  console.log('Python backend available:', useBackend);

  if (!useBackend && !textureExtractor) {
    console.error('No face extraction method available');
    status.textContent = 'Error: Face extractor not available. Start Python server.';
    return;
  }

  status.textContent = useBackend ? 'Using Python face extractor...' : 'Initializing face detection...';
  console.log('Starting face detection...');
  
  try {
    // Initialize JS texture extractor if needed and backend not available
    if (!useBackend) {
      console.log('Calling textureExtractor.initialize()...');
      await textureExtractor.initialize();
      console.log('textureExtractor initialized successfully');
    }
    
    // Clear previous textures
    textureBlender.clear();
    
    let processedCount = 0;
    
    for (const entry of uploadedImages) {
      if (entry.processed) continue;
      
      entry.previewElement?.classList.add('processing');
      status.textContent = `Processing image ${processedCount + 1}/${uploadedImages.length}...`;
      
      try {
        let texture;
        
        // Always use JS texture extractor for UV mapping (uses MediaPipe 468 landmarks)
        // The Python backend gives better face detection but only 5 landmarks
        // So we use the original image with JS extractor for proper UV mapping
        console.log('Using JS texture extractor for UV mapping...');
        
        // Initialize if not already done
        if (!textureExtractor.initialized) {
          console.log('Initializing texture extractor...');
          await textureExtractor.initialize();
        }
        
        // Extract texture with proper UV mapping
        const result = await textureExtractor.extractTexture(entry.img);
        texture = result.texture;
        const landmarks = result.landmarks;
        
        // Detect head pose from landmarks
        const detectedAngle = textureExtractor.detectHeadPose(landmarks);
        console.log('Detected head pose:', detectedAngle, 'for', entry.file.name);
        
        // Display extracted texture for debugging
        const extractedContainer = document.getElementById('extractedFacesContainer');
        const extractedSection = document.getElementById('extractedFaces');
        if (extractedContainer && extractedSection) {
          extractedSection.style.display = 'block';
          const wrapper = document.createElement('div');
          wrapper.style.cssText = 'text-align: center;';
          const label = document.createElement('p');
          label.textContent = `${entry.file.name} (${detectedAngle})`;
          label.style.cssText = 'font-size: 10px; color: rgba(255,255,255,0.5); margin-bottom: 4px; max-width: 100px; overflow: hidden; text-overflow: ellipsis;';
          const displayCanvas = document.createElement('canvas');
          displayCanvas.width = 100;
          displayCanvas.height = 100;
          displayCanvas.style.cssText = 'border: 1px solid rgba(0, 217, 255, 0.5); border-radius: 4px;';
          const displayCtx = displayCanvas.getContext('2d');
          displayCtx.drawImage(texture, 0, 0, 100, 100);
          wrapper.appendChild(label);
          wrapper.appendChild(displayCanvas);
          extractedContainer.appendChild(wrapper);
        }
        
        // Add to blender with detected angle and landmarks
        textureBlender.addTexture(texture, 1.0, detectedAngle, landmarks);
        
        entry.processed = true;
        entry.previewElement?.classList.remove('processing');
        entry.previewElement?.classList.add('processed');
        processedCount++;
        
        console.log('Processed image:', entry.file.name, useBackend ? '(backend)' : '(js)');
        
        // Compare extraction methods for the first image
        if (processedCount === 1 && useBackend) {
          try {
            // Convert image to base64 for comparison
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = entry.img.width;
            tempCanvas.height = entry.img.height;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(entry.img, 0, 0);
            const imageDataUrl = tempCanvas.toDataURL('image/png');
            await compareExtractionMethods(imageDataUrl);
          } catch (compareErr) {
            console.log('Comparison skipped:', compareErr.message);
          }
        }
      } catch (err) {
        console.error('Failed to process image:', entry.file.name, err);
        entry.previewElement?.classList.remove('processing');
        status.textContent = `Failed to detect face in ${entry.file.name}`;
      }
    }
    
    if (processedCount === 0) {
      status.textContent = 'No faces detected in any image';
      return;
    }
    
    // Blend all textures
    status.textContent = 'Blending textures...';
    faceSwapTexture = textureBlender.blend();
    
    // Analyze source face colors
    const sourceStats = colorMatcher.analyzeColors(faceSwapTexture);
    
    // Show preview
    if (texturePreview && textureCanvas) {
      texturePreview.style.display = 'block';
      const previewCtx = textureCanvas.getContext('2d');
      textureCanvas.width = 200;
      textureCanvas.height = 200;
      previewCtx.drawImage(faceSwapTexture, 0, 0, 200, 200);
    }
    
    // Load into WebGL renderer
    if (webglMaskRenderer && faceSwapTexture) {
      webglMaskRenderer.loadFaceSwapTexture(faceSwapTexture);
    }
    
    // Enable face swap mode
    faceSwapMode = true;
    toggleFaceSwapBtn?.classList.add('active');
    
    status.textContent = `Face swap ready! Processed ${processedCount} image(s)`;
    status.style.color = '#00ff88';
    
  } catch (err) {
    console.error('Processing error:', err);
    status.textContent = 'Error processing images: ' + err.message;
    status.style.color = '#ff6b6b';
  }
}

// Clear all uploaded images
function clearUploadedImages() {
  uploadedImages = [];
  if (previewContainer) {
    previewContainer.innerHTML = '';
  }
  if (texturePreview) {
    texturePreview.style.display = 'none';
  }
  // Clear extracted faces debug display
  const extractedContainer = document.getElementById('extractedFacesContainer');
  const extractedSection = document.getElementById('extractedFaces');
  if (extractedContainer) extractedContainer.innerHTML = '';
  if (extractedSection) extractedSection.style.display = 'none';
  
  // Clear comparison display
  const comparisonContainer = document.getElementById('methodComparisonContainer');
  const comparisonSection = document.getElementById('methodComparison');
  if (comparisonContainer) comparisonContainer.innerHTML = '';
  if (comparisonSection) comparisonSection.style.display = 'none';
  
  faceSwapTexture = null;
  faceSwapMode = false;
  toggleFaceSwapBtn?.classList.remove('active');
  textureBlender?.clear();
  
  // Reload procedural mask
  if (webglMaskRenderer) {
    webglMaskRenderer.loadProceduralMask();
  }
  
  status.textContent = 'Cleared all images';
}

// Compare extraction methods from backend
async function compareExtractionMethods(imageData) {
  const comparisonContainer = document.getElementById('methodComparisonContainer');
  const comparisonSection = document.getElementById('methodComparison');
  
  if (!comparisonContainer || !comparisonSection) return;
  
  try {
    const response = await fetch('http://localhost:5000/extract-compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData })
    });
    
    if (!response.ok) {
      console.log('Comparison endpoint not available');
      return;
    }
    
    const data = await response.json();
    
    if (data.success && data.results) {
      comparisonSection.style.display = 'block';
      comparisonContainer.innerHTML = '';
      
      const methods = [
        { id: 'retinaface', name: 'RetinaFace', color: '#00d9ff' },
        { id: 'face_alignment', name: 'Face Align 3D', color: '#00ff88' },
        { id: 'nextface', name: 'NextFace', color: '#ff9500' },
        { id: 'gantex', name: '3D-GANTex', color: '#af52de' }
      ];
      
      for (const method of methods) {
        const result = data.results[method.id];
        const wrapper = document.createElement('div');
        wrapper.style.cssText = `
          text-align: center;
          padding: 10px;
          background: rgba(255,255,255,0.03);
          border-radius: 8px;
          border: 1px solid ${result?.success ? method.color : 'rgba(255,255,255,0.1)'};
          opacity: ${result?.success ? '1' : '0.5'};
          min-width: 120px;
        `;
        
        const label = document.createElement('p');
        label.textContent = method.name;
        label.style.cssText = `font-size: 11px; color: ${method.color}; margin-bottom: 8px; font-weight: bold;`;
        wrapper.appendChild(label);
        
        if (result?.success && result.face) {
          const img = document.createElement('img');
          img.src = result.face;
          img.style.cssText = 'width: 100px; height: 100px; object-fit: cover; border-radius: 4px;';
          wrapper.appendChild(img);
          
          // Add "Use this" button
          const useBtn = document.createElement('button');
          useBtn.textContent = 'Use';
          useBtn.style.cssText = `
            margin-top: 8px; padding: 4px 12px; font-size: 10px;
            background: ${method.color}33; border: 1px solid ${method.color};
            color: white; border-radius: 4px; cursor: pointer;
          `;
          useBtn.onclick = () => selectExtractionMethod(method.id, result.face);
          wrapper.appendChild(useBtn);
        } else {
          const errorText = document.createElement('p');
          errorText.textContent = result?.error || 'Not available';
          errorText.style.cssText = 'font-size: 10px; color: rgba(255,255,255,0.4); padding: 20px 0;';
          wrapper.appendChild(errorText);
        }
        
        comparisonContainer.appendChild(wrapper);
      }
    }
  } catch (err) {
    console.log('Comparison fetch failed:', err.message);
  }
}

// Select a specific extraction method result
function selectExtractionMethod(methodId, faceDataUrl) {
  console.log('Selected extraction method:', methodId);
  status.textContent = `Using ${methodId} extraction`;
  
  // Load the selected face as the swap texture
  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 512, 512);
    
    faceSwapTexture = canvas;
    
    // Update preview
    if (texturePreview && textureCanvas) {
      texturePreview.style.display = 'block';
      const previewCtx = textureCanvas.getContext('2d');
      textureCanvas.width = 200;
      textureCanvas.height = 200;
      previewCtx.drawImage(canvas, 0, 0, 200, 200);
    }
    
    // Load into WebGL
    if (webglMaskRenderer) {
      webglMaskRenderer.loadFaceSwapTexture(canvas);
    }
    
    faceSwapMode = true;
    toggleFaceSwapBtn?.classList.add('active');
    status.textContent = `Face swap ready (${methodId})`;
    status.style.color = '#00ff88';
  };
  img.src = faceDataUrl;
}

// Current expressions state (for use by renderers)
let currentExpressions = null;

// FPS tracking
let lastFrameTime = performance.now();
let frameCount = 0;
let currentFPS = 0;

// Handle face detection results
function onFaceResults(results) {
  // Update FPS counter
  frameCount++;
  const now = performance.now();
  if (now - lastFrameTime >= 1000) {
    currentFPS = frameCount;
    frameCount = 0;
    lastFrameTime = now;
  }
  
  // Clear both canvases
  maskRenderer.clear();
  if (webglMaskRenderer) {
    webglMaskRenderer.clear();
  }

  if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
    const rawLandmarks = results.multiFaceLandmarks[0];
    
    let landmarks = transformLandmarksSimple(rawLandmarks, window.videoTransform, video);
    
    // Store current landmarks for click detection in connection mode
    currentLandmarks = landmarks;
    
    // Apply face morphing if enabled
    if (morphSettings.enabled) {
      landmarks = maskRenderer.drawMorphedFace(landmarks, canvas.width, canvas.height, morphSettings) || landmarks;
    }
    
    // Store expressions for use by renderers
    currentExpressions = results.expressions;

    // Get face transform for 2D operations
    const transform = getFaceTransform(landmarks, canvas.width, canvas.height);

    // Draw mesh visualization (2D canvas)
    if (showMesh && !faceSwapMode) {
      // Draw triangle mesh with 3D depth visualization
      // Apply global opacity
      maskRenderer.ctx.globalAlpha = meshSettings.meshOpacity;
      
      // Apply pulse effect if enabled
      let pulseScale = 1;
      if (meshSettings.pulseEffect) {
        const time = performance.now() * 0.001 * meshSettings.animationSpeed;
        pulseScale = 1 + Math.sin(time * 3) * 0.02;
      }
      
      maskRenderer.drawTriangleMesh(landmarks, canvas.width, canvas.height, {
        lineWidth: meshSettings.lineWidth * pulseScale,
        showVertices: meshSettings.showVertices,
        vertexRadius: meshSettings.vertexRadius * pulseScale,
        useDepth: meshSettings.useDepth,
        showContours: meshSettings.showContours,
        showTriangles: meshSettings.showTriangles,
        strokeColor: meshSettings.strokeColor,
        fillColor: meshSettings.fillColor,
        fillOpacity: meshSettings.fillOpacity,
        // Adaptive LOD settings
        useAdaptiveLOD: meshSettings.useAdaptiveLOD,
        enableDenseLandmarks: meshSettings.enableDenseLandmarks,
        enableSubdivision: meshSettings.enableSubdivision,
        lodLevel: meshSettings.lodLevel,
        enableSymmetry: meshSettings.enableSymmetry,
        showRegionColors: meshSettings.showRegionColors,
        // Contour settings
        contourWidth: meshSettings.contourWidth,
        contourColor: meshSettings.contourColor,
        showEyes: meshSettings.showEyes,
        showEyebrows: meshSettings.showEyebrows,
        showLips: meshSettings.showLips,
        showNose: meshSettings.showNose,
        showFaceOval: meshSettings.showFaceOval
      });
      
      // Reset opacity
      maskRenderer.ctx.globalAlpha = 1;
      
      // Draw key reference points
      if (meshSettings.showKeyPoints) {
        maskRenderer.drawKeyPoints(transform);
      }
      
      // Draw iris tracking (landmarks 468-477)
      if (meshSettings.showIrisTracking) {
        maskRenderer.drawIrisTracking(landmarks, canvas.width, canvas.height, {
          showGazeDirection: meshSettings.showGazeDirection
        });
      }
      
      // Draw expression-reactive effects (labels)
      if (meshSettings.showExpressions && results.expressions) {
        maskRenderer.drawExpressionEffects(landmarks, canvas.width, canvas.height, results.expressions);
      }
      
      // Draw mood detection (valence-arousal analysis)
      if (meshSettings.showMoodDetection && results.expressions && results.expressions.raw) {
        // Analyze mood from raw blendshape scores
        const moodData = moodAnalyzer.analyze(results.expressions.raw);
        
        // Draw mood indicator
        maskRenderer.drawMoodIndicator(moodData, canvas.width, canvas.height, {
          showMoodLabel: true,
          showEmotionWheel: meshSettings.showEmotionWheel,
          showEmotionBars: meshSettings.showEmotionBars,
          position: 'top-left'  // Put mood on left, expressions on right
        });
      }
      
      // Draw landmark indices if enabled
      if (meshSettings.showLandmarkIndices) {
        maskRenderer.drawLandmarkIndices(landmarks, canvas.width, canvas.height);
      }
      
      // Draw makeup (applies to mesh view too)
      if (makeupSettings.enabled) {
        maskRenderer.drawMakeup(landmarks, canvas.width, canvas.height, makeupSettings);
      }
      
      // Draw accessories
      if (accessorySettings.enabled && accessorySettings.type !== 'none') {
        maskRenderer.drawAccessory(landmarks, canvas.width, canvas.height, accessorySettings);
      }
      
      // Draw custom connections
      if (customConnections.length > 0) {
        maskRenderer.drawCustomConnections(landmarks, canvas.width, canvas.height, customConnections);
      }
      
      // Highlight selected landmark in connection mode
      if (isConnectionMode && selectedLandmark !== null) {
        maskRenderer.highlightLandmark(landmarks, canvas.width, canvas.height, selectedLandmark);
      }
    }
    
    // Hand tracking (runs independently of face detection)
    if (meshSettings.enableHandTracking && handTracker) {
      const timestamp = performance.now();
      const handResults = handTracker.detect(timestamp);
      
      if (handResults && handResults.count > 0) {
        maskRenderer.drawHands(handResults, canvas.width, canvas.height, {
          showConnections: meshSettings.showHandConnections,
          showLandmarks: meshSettings.showHandLandmarks,
          showGesture: meshSettings.showHandGestures,
          showLabels: meshSettings.showHandLabels,
          mirrorMode: meshSettings.mirrorVideo
        });
      }
    }

    // Face swap mode - use extracted face texture
    if (faceSwapMode && faceSwapTexture && webglMaskRenderer && webglMaskRenderer.maskLoaded) {
      // Draw face swap with edge feathering
      webglMaskRenderer.drawMask(landmarks, canvas.width, canvas.height, 0.95, 0.9);
      
      const modeText = 'Face Swap Active';
      status.textContent = `${modeText} (${landmarks.length} landmarks)`;
    }
    // Regular mask mode - use custom procedural mask
    else if (showMask) {
      // Draw custom procedural mask using current design settings
      maskRenderer.drawCustomMask(landmarks, canvas.width, canvas.height, maskDesign);
      
      // Draw makeup on top of mask
      if (makeupSettings.enabled) {
        maskRenderer.drawMakeup(landmarks, canvas.width, canvas.height, makeupSettings);
      }
      
      // Draw accessories
      if (accessorySettings.enabled && accessorySettings.type !== 'none') {
        maskRenderer.drawAccessory(landmarks, canvas.width, canvas.height, accessorySettings);
      }
      
      status.textContent = `Mask: ${maskDesign.shape} (${landmarks.length} landmarks)`;
    }
    else {
      // Build expression status string
      const expressionText = buildExpressionStatus(results.expressions);
      const fpsText = meshSettings.showFPS ? ` | ${currentFPS} FPS` : '';
      status.textContent = `Tracking face (${landmarks.length} landmarks)${expressionText}${fpsText}`;
    }
    
    // Draw FPS overlay if enabled
    if (meshSettings.showFPS) {
      maskRenderer.drawFPS(currentFPS);
    }
  } else {
    const fpsText = meshSettings.showFPS ? ` | ${currentFPS} FPS` : '';
    status.textContent = `No face detected${fpsText}`;
    currentExpressions = null;
    
    if (meshSettings.showFPS) {
      maskRenderer.drawFPS(currentFPS);
    }
  }
  
  // Update recording canvas if recording
  if (isRecording) {
    updateRecordingCanvas();
  }
}

// Build a status string showing detected expressions
function buildExpressionStatus(expressions) {
  if (!expressions) return '';
  
  const detected = [];
  
  if (expressions.smiling) detected.push('😊');
  if (expressions.mouthOpen) detected.push('😮');
  if (expressions.bothEyesClosed) detected.push('😑');
  else if (expressions.leftEyeClosed || expressions.rightEyeClosed) detected.push('😉');
  if (expressions.surprised) detected.push('😲');
  if (expressions.cheeksPuffed) detected.push('🐡');
  if (expressions.mouthPucker) detected.push('😗');
  
  if (detected.length > 0) {
    return ' ' + detected.join('');
  }
  return '';
}

// Draw a simple 2D face mask using canvas
function draw2DFaceMask(ctx, landmarks, width, height) {
  if (!landmarks || landmarks.length < 468) return;
  
  // Get face outline points
  const faceOutline = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
  ];
  
  // Outer lips
  const lipsOuter = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
    409, 270, 269, 267, 0, 37, 39, 40, 185
  ];
  
  // Inner lips (for the mouth opening)
  const lipsInner = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191
  ];
  
  ctx.save();
  
  // Draw filled face shape
  ctx.beginPath();
  const first = landmarks[faceOutline[0]];
  ctx.moveTo(first.x * width, first.y * height);
  
  for (let i = 1; i < faceOutline.length; i++) {
    const lm = landmarks[faceOutline[i]];
    ctx.lineTo(lm.x * width, lm.y * height);
  }
  ctx.closePath();
  
  // Semi-transparent fill
  ctx.fillStyle = 'rgba(0, 200, 255, 0.3)';
  ctx.fill();
  
  // Border
  ctx.strokeStyle = 'rgba(0, 255, 200, 0.6)';
  ctx.lineWidth = 2;
  ctx.stroke();
  
  // Draw lips/mouth overlay with different color
  ctx.beginPath();
  const lipFirst = landmarks[lipsOuter[0]];
  ctx.moveTo(lipFirst.x * width, lipFirst.y * height);
  
  for (let i = 1; i < lipsOuter.length; i++) {
    const lm = landmarks[lipsOuter[i]];
    ctx.lineTo(lm.x * width, lm.y * height);
  }
  ctx.closePath();
  
  // Lips fill - slightly different color
  ctx.fillStyle = 'rgba(255, 100, 150, 0.4)';
  ctx.fill();
  
  // Lips border
  ctx.strokeStyle = 'rgba(255, 150, 180, 0.7)';
  ctx.lineWidth = 1.5;
  ctx.stroke();
  
  // Draw inner mouth (darker area)
  ctx.beginPath();
  const innerFirst = landmarks[lipsInner[0]];
  ctx.moveTo(innerFirst.x * width, innerFirst.y * height);
  
  for (let i = 1; i < lipsInner.length; i++) {
    const lm = landmarks[lipsInner[i]];
    ctx.lineTo(lm.x * width, lm.y * height);
  }
  ctx.closePath();
  
  // Inner mouth - darker
  ctx.fillStyle = 'rgba(80, 20, 40, 0.5)';
  ctx.fill();
  
  ctx.restore();
}

// Start the app
init();
