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

// Face swap components
let textureExtractor = null;
let textureBlender = null;
let colorMatcher = null;
let uploadedImages = [];  // Array of { file, img, processed: boolean }
let faceSwapTexture = null;  // The final blended face texture
let lastLiveFaceStats = null;  // Color stats from live video

// Initialize the app
async function init() {
  console.log('Starting initialization...');
  status.textContent = 'Requesting camera...';
  
  // Request camera FIRST - this is the most important step
  try {
    console.log('Calling getUserMedia...');
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 360, height: 640, facingMode: 'user' }
    });
    console.log('Got camera stream:', stream);
    video.srcObject = stream;
    await video.play();
    console.log('Camera playing');
    status.textContent = 'Camera active!';
  } catch (camError) {
    console.error('Camera error:', camError);
    status.textContent = 'Camera error: ' + camError.message;
    status.style.color = '#ff6b6b';
    return;
  }

  try {
    // Set canvas size (9:16 ratio)
    canvas.width = 360;
    canvas.height = 640;

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
    webglCanvas.width = 360;
    webglCanvas.height = 640;
    // Use same styling as the 2D canvas
    webglCanvas.style.position = 'absolute';
    webglCanvas.style.top = '0';
    webglCanvas.style.left = '0';
    webglCanvas.style.transform = 'scaleX(-1)';
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

    // Create face tracker
    status.textContent = 'Loading face mesh model...';
    console.log('Creating face tracker...');
    faceTracker = new FaceTracker(video, onFaceResults);

    await faceTracker.initialize();
    console.log('Face tracker initialized');

    status.textContent = 'Ready! 468 landmarks active';
    status.style.color = '#00ff88';

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

// Handle face detection results
function onFaceResults(results) {
  // Clear both canvases
  maskRenderer.clear();
  if (webglMaskRenderer) {
    webglMaskRenderer.clear();
  }

  if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
    const landmarks = results.multiFaceLandmarks[0];

    // Get face transform for 2D operations
    const transform = getFaceTransform(landmarks, canvas.width, canvas.height);

    // Draw mesh visualization (2D canvas)
    if (showMesh && !faceSwapMode) {
      maskRenderer.drawMesh(landmarks, canvas.width, canvas.height);
      maskRenderer.drawKeyPoints(transform);
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
      status.textContent = `Tracking face (${landmarks.length} landmarks)`;
    }
  } else {
    status.textContent = 'No face detected';
  }
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
