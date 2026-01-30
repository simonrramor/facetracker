// Face Filter App - Main Entry Point
import { FaceTracker, getFaceTransform } from './face-tracker.js';
import { MaskRenderer } from './mask-renderer.js';
import { WebGLMaskRenderer } from './webgl-mask-renderer.js';
import { FaceTextureExtractor, loadImageFromFile, loadImageFromURL } from './face-texture-extractor.js';
import { TextureBlender } from './texture-blender.js';
import { ColorMatcher } from './color-matcher.js';

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
let maskRenderer = null;  // 2D canvas renderer for mesh visualization
let webglMaskRenderer = null;  // WebGL renderer for 3D mask wrapping

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
  // Contours
  contourWidth: 1.5,
  contourColor: '#00ff88',
  showEyes: true,
  showEyebrows: true,
  showLips: true,
  showNose: true,
  showFaceOval: true,
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

// Calculate transform for object-fit: cover
// Video is 1920x1080 (landscape), display is 378x756 (portrait)
// With cover, video height fills display, sides are cropped
function calculateVideoTransform(videoWidth, videoHeight, displayWidth, displayHeight) {
  const videoAspect = videoWidth / videoHeight;  // 1.78 (landscape)
  const displayAspect = displayWidth / displayHeight;  // 0.5 (portrait)
  
  // Video is wider - scaled by height, sides cropped
  // Scaled video width in display pixels
  const scaledVideoWidth = displayHeight * videoAspect;  // 756 * 1.78 = 1345
  
  // How much is cropped from each side (in video normalized coords)
  const cropX = (scaledVideoWidth - displayWidth) / scaledVideoWidth / 2;  // 0.36
  
  // Visible portion of video X (normalized)
  const visibleXStart = cropX;  // 0.36
  const visibleXEnd = 1 - cropX;  // 0.64
  const visibleXRange = visibleXEnd - visibleXStart;  // 0.28
  
  return {
    videoWidth, videoHeight,
    displayWidth, displayHeight,
    visibleXStart,
    visibleXRange
  };
}

// Transform landmarks with crop adjustment AND mirroring
// SOLUTION 3: Handle mirroring in JavaScript instead of CSS
function transformLandmarksSimple(landmarks, transform) {
  if (!transform) return landmarks;
  
  const { visibleXStart, visibleXRange } = transform;
  
  return landmarks.map(lm => {
    // Map X from visible range to [0, 1]
    let x = (lm.x - visibleXStart) / visibleXRange;
    
    // Mirror X coordinate (replaces CSS scaleX(-1))
    x = 1 - x;
    
    const y = lm.y;
    
    return { x, y, z: lm.z };
  });
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
    { id: 'contourWidth', key: 'contourWidth', format: v => v.toFixed(1) },
    { id: 'animationSpeed', key: 'animationSpeed', format: v => v.toFixed(1) + 'x' },
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
      });
    }
  });

  // Toggle controls
  const toggles = [
    'showVertices', 'showContours', 'useDepth', 'showTriangles',
    'showEyes', 'showEyebrows', 'showLips', 'showNose', 'showFaceOval',
    'showExpressions', 'showKeyPoints', 'pulseEffect',
    'mirrorVideo', 'showVideo', 'showFPS', 'showLandmarkIndices'
  ];
  
  toggles.forEach(key => {
    const toggle = document.getElementById(key);
    if (toggle) {
      toggle.checked = meshSettings[key];
      toggle.addEventListener('change', (e) => {
        meshSettings[key] = e.target.checked;
        
        // Apply special effects
        if (key === 'mirrorVideo') {
          video.style.transform = e.target.checked ? 'scaleX(-1)' : 'none';
        }
        if (key === 'showVideo') {
          video.style.opacity = e.target.checked ? '1' : '0';
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
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    }
  });
}

// Apply video CSS filters
function applyVideoFilters() {
  const { videoBrightness, videoContrast, videoSaturation } = meshSettings;
  video.style.filter = `brightness(${videoBrightness}) contrast(${videoContrast}) saturate(${videoSaturation})`;
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
    { id: 'contourWidth', format: v => v.toFixed(1) },
    { id: 'animationSpeed', format: v => v.toFixed(1) + 'x' },
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
    'showEyes', 'showEyebrows', 'showLips', 'showNose', 'showFaceOval',
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
    // SOLUTION 3: Canvas at display size, mirroring in JavaScript
    const displayWidth = 378;
    const displayHeight = 756;
    canvas.width = displayWidth;
    canvas.height = displayHeight;
    console.log('Canvas set to display size:', canvas.width, 'x', canvas.height);
    console.log('Video actual size:', video.videoWidth, 'x', video.videoHeight);
    
    // Calculate coordinate transformation for object-fit: cover
    window.videoTransform = calculateVideoTransform(
      video.videoWidth, video.videoHeight,
      displayWidth, displayHeight
    );
    console.log('Video transform:', window.videoTransform);

    // Create 2D mask renderer for mesh visualization
    maskRenderer = new MaskRenderer(canvas);
    console.log('2D mask renderer created');

    // Create WebGL canvas for 3D mask rendering
    status.textContent = 'Setting up WebGL...';
    const videoWrap = document.querySelector('.video-wrap');
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
    webglCanvas.width = displayWidth;
    webglCanvas.height = displayHeight;
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
      status.textContent = 'Ready! 478 landmarks + 52 expressions';
      status.style.color = '#00ff88';
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

    // Initialize settings controls
    initSettingsControls();

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
    // SOLUTION 3: Transform landmarks with crop adjustment AND mirroring in JS
    const rawLandmarks = results.multiFaceLandmarks[0];
    const landmarks = transformLandmarksSimple(rawLandmarks, window.videoTransform);
    
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
      
      // Draw expression-reactive effects
      if (meshSettings.showExpressions && results.expressions) {
        maskRenderer.drawExpressionEffects(landmarks, canvas.width, canvas.height, results.expressions);
      }
      
      // Draw landmark indices if enabled
      if (meshSettings.showLandmarkIndices) {
        maskRenderer.drawLandmarkIndices(landmarks, canvas.width, canvas.height);
      }
    }

    // Face swap mode - use extracted face texture
    if (faceSwapMode && faceSwapTexture && webglMaskRenderer && webglMaskRenderer.maskLoaded) {
      // Draw face swap with edge feathering
      webglMaskRenderer.drawMask(landmarks, canvas.width, canvas.height, 0.95, 0.9);
      
      const modeText = 'Face Swap Active';
      status.textContent = `${modeText} (${landmarks.length} landmarks)`;
    }
    // Regular mask mode
    else if (showMask) {
      if (webglMaskRenderer && webglMaskRenderer.maskLoaded) {
        webglMaskRenderer.drawMask(landmarks, canvas.width, canvas.height, 0.6, 0.5);
      }
      // Also draw 2D mask as overlay
      draw2DFaceMask(maskRenderer.ctx, landmarks, canvas.width, canvas.height);
      
      status.textContent = `Tracking face (${landmarks.length} landmarks)`;
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
