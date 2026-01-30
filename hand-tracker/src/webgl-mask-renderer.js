// WebGL Mask Renderer - Renders mask textures onto 3D face mesh
// Uses actual face mesh geometry for proper wrapping/warping

// MediaPipe canonical face mesh UVs and triangle indices
// These define the topology of the face mesh
import { FACE_MESH_TRIANGULATION, FACE_MESH_UVS } from './face-mesh-data.js';

export class WebGLMaskRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl', { 
      alpha: true, 
      premultipliedAlpha: false,
      antialias: true 
    });
    
    if (!this.gl) {
      throw new Error('WebGL not supported');
    }
    
    this.program = null;
    this.maskTexture = null;
    this.maskLoaded = false;
    this.positionBuffer = null;
    this.texCoordBuffer = null;
    this.indexBuffer = null;
    
    // Which region of the face to render the mask on
    // These are indices into the triangulation that cover the upper face
    this.maskTriangleIndices = null;
    
    // Generate procedural mask texture as fallback
    this.proceduralMaskCanvas = null;
    
    this.init();
  }
  
  // Generate a procedural full face mask texture (semi-transparent)
  generateProceduralMask() {
    const size = 512;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    
    // Fill with a visible semi-transparent color
    // Using bright magenta/pink so it's very visible for debugging
    ctx.fillStyle = 'rgba(255, 0, 150, 0.5)';
    ctx.fillRect(0, 0, size, size);
    
    console.log('Generated procedural mask texture:', size, 'x', size);
    
    this.proceduralMaskCanvas = canvas;
    return canvas;
  }
  
  init() {
    const gl = this.gl;
    
    // Vertex shader - transforms 2D face mesh vertices
    // Also passes normalized position for edge blending
    const vsSource = `
      attribute vec2 a_position;
      attribute vec2 a_texCoord;
      
      uniform vec2 u_resolution;
      
      varying vec2 v_texCoord;
      varying vec2 v_uvPosition;
      
      void main() {
        // Convert from pixels to clip space (-1 to 1)
        vec2 clipSpace = ((a_position / u_resolution) * 2.0 - 1.0) * vec2(1, -1);
        gl_Position = vec4(clipSpace, 0, 1);
        v_texCoord = a_texCoord;
        v_uvPosition = a_texCoord; // Pass UV for edge calculations
      }
    `;
    
    // Fragment shader - samples mask texture with edge blending
    const fsSource = `
      precision mediump float;
      
      uniform sampler2D u_texture;
      uniform float u_opacity;
      uniform float u_edgeFeather; // 0.0 = sharp edges, 1.0 = very soft edges
      
      varying vec2 v_texCoord;
      varying vec2 v_uvPosition;
      
      void main() {
        // Clamp UV coordinates to valid range
        vec2 uv = clamp(v_texCoord, 0.0, 1.0);
        vec4 color = texture2D(u_texture, uv);
        
        // Only apply if texture has content (alpha > 0)
        if (color.a < 0.01) {
          discard;
        }
        
        // Simple edge fade based on distance from center
        vec2 center = vec2(0.5, 0.5);
        float distFromCenter = distance(v_uvPosition, center);
        
        // Gentle edge fade - only fade at very edges
        float edgeFade = 1.0 - smoothstep(0.4, 0.55, distFromCenter);
        
        // Apply feathering based on uniform
        float finalFade = mix(1.0, edgeFade, u_edgeFeather);
        
        gl_FragColor = vec4(color.rgb, color.a * u_opacity * finalFade);
      }
    `;
    
    // Compile shaders
    const vertexShader = this.compileShader(gl.VERTEX_SHADER, vsSource);
    const fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, fsSource);
    
    // Create program
    this.program = gl.createProgram();
    gl.attachShader(this.program, vertexShader);
    gl.attachShader(this.program, fragmentShader);
    gl.linkProgram(this.program);
    
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error('Program link failed: ' + gl.getProgramInfoLog(this.program));
    }
    
    // Get attribute/uniform locations
    this.locations = {
      position: gl.getAttribLocation(this.program, 'a_position'),
      texCoord: gl.getAttribLocation(this.program, 'a_texCoord'),
      resolution: gl.getUniformLocation(this.program, 'u_resolution'),
      texture: gl.getUniformLocation(this.program, 'u_texture'),
      opacity: gl.getUniformLocation(this.program, 'u_opacity'),
      edgeFeather: gl.getUniformLocation(this.program, 'u_edgeFeather')
    };
    
    console.log('WebGL locations:', {
      position: this.locations.position,
      texCoord: this.locations.texCoord,
      resolution: this.locations.resolution !== null,
      texture: this.locations.texture !== null,
      opacity: this.locations.opacity !== null,
      edgeFeather: this.locations.edgeFeather !== null
    });
    
    // Create buffers
    this.positionBuffer = gl.createBuffer();
    this.texCoordBuffer = gl.createBuffer();
    this.indexBuffer = gl.createBuffer();
    
    // Set up tex coord buffer with canonical UVs
    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(FACE_MESH_UVS), gl.STATIC_DRAW);
    
    // Set up index buffer with triangulation
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(FACE_MESH_TRIANGULATION), gl.STATIC_DRAW);
    
    // Pre-compute mask region triangles (upper face for superhero mask)
    this.computeMaskRegion();
    
    // Enable blending for transparency
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }
  
  compileShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error('Shader compile failed: ' + info);
    }
    
    return shader;
  }
  
  // Compute which triangles belong to the mask region (upper face)
  computeMaskRegion() {
    // Upper face landmark indices for superhero mask
    // Covers forehead, eyebrows, eyes, and nose bridge
    const upperFaceIndices = new Set([
      // Forehead
      10, 151, 9, 8, 168, 6, 197, 195, 5,
      // Around left eye
      33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
      70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
      // Around right eye  
      362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
      300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
      // Nose bridge
      168, 6, 197, 195, 5, 4, 1,
      // Eye socket areas
      156, 35, 31, 228, 229, 230, 231, 232, 233, 244, 143, 111, 117, 118, 119, 120, 121, 128,
      383, 265, 261, 448, 449, 450, 451, 452, 453, 464, 372, 340, 346, 347, 348, 349, 350, 357,
      // Upper cheek/temple
      234, 127, 162, 21, 54, 103, 67, 109,
      454, 356, 389, 251, 284, 332, 297, 338,
      // Fill in upper face area
      69, 68, 104, 43, 48, 131, 134, 51, 49, 220, 45, 4, 
      299, 298, 333, 273, 278, 360, 363, 281, 279, 440, 275,
      // More complete coverage
      71, 175, 226, 25, 110, 24, 23, 22, 26, 112, 243, 190, 56, 28, 27, 29, 30,
      301, 399, 446, 255, 339, 254, 253, 252, 256, 341, 463, 414, 286, 258, 257, 259, 260,
      // Inner eye corners and nose bridge
      189, 221, 222, 223, 224, 225, 113, 130, 247,
      413, 441, 442, 443, 444, 445, 342, 359, 467
    ]);
    
    // Find triangles where at least 2 vertices are in the upper face
    this.maskTriangleIndices = [];
    const triangulation = FACE_MESH_TRIANGULATION;
    
    for (let i = 0; i < triangulation.length; i += 3) {
      const v0 = triangulation[i];
      const v1 = triangulation[i + 1];
      const v2 = triangulation[i + 2];
      
      const count = (upperFaceIndices.has(v0) ? 1 : 0) +
                    (upperFaceIndices.has(v1) ? 1 : 0) +
                    (upperFaceIndices.has(v2) ? 1 : 0);
      
      // Include triangle if at least 2 vertices in mask region
      if (count >= 2) {
        this.maskTriangleIndices.push(v0, v1, v2);
      }
    }
    
    console.log(`Mask region: ${this.maskTriangleIndices.length / 3} triangles`);
  }
  
  async loadMask(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      img.onload = () => {
        this.createTextureFromImage(img);
        console.log('WebGL mask texture loaded:', img.width, 'x', img.height);
        resolve(img);
      };
      
      img.onerror = reject;
      img.src = url;
    });
  }
  
  // Load the procedural mask as fallback
  loadProceduralMask() {
    const canvas = this.generateProceduralMask();
    this.createTextureFromImage(canvas);
    console.log('WebGL procedural mask texture created');
  }
  
  createTextureFromImage(imageSource) {
    const gl = this.gl;
    
    // Create texture
    this.maskTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
    
    // Upload image to texture (no Y flip - we handle coords directly)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imageSource);
    
    // Set texture parameters
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    
    this.maskLoaded = true;
  }
  
  clear() {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }
  
  // Debug: draw a test rectangle at center of canvas
  drawTestRect() {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    
    // Simple colored rectangle using basic WebGL
    const vertices = new Float32Array([
      // Center rectangle (200x200)
      220, 140,
      420, 140,
      420, 340,
      220, 340
    ]);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.DYNAMIC_DRAW);
    
    gl.useProgram(this.program);
    gl.enableVertexAttribArray(this.locations.position);
    gl.vertexAttribPointer(this.locations.position, 2, gl.FLOAT, false, 0, 0);
    
    gl.uniform2f(this.locations.resolution, this.canvas.width, this.canvas.height);
    gl.uniform1f(this.locations.opacity, 1.0);
    gl.uniform1f(this.locations.edgeFeather, 0.0);
    
    // Draw as triangle fan
    gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
    console.log('Drew test rectangle at canvas center');
  }
  
  // Render mask onto face mesh (full face)
  // edgeFeather: 0.0 = sharp edges, 1.0 = soft feathered edges
  drawMask(landmarks, canvasWidth, canvasHeight, opacity = 0.5, edgeFeather = 0.8) {
    if (!this.maskLoaded) {
      console.log('drawMask: mask not loaded');
      return;
    }
    if (!landmarks || landmarks.length < 468) {
      console.log('drawMask: not enough landmarks:', landmarks?.length);
      return;
    }
    
    const gl = this.gl;
    
    // Set viewport to match canvas
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    
    // Only use first 468 landmarks (the base mesh)
    const numVertices = 468;
    
    // Get key anchor landmarks from live face
    const leftEye = landmarks[33];   // Left eye outer
    const rightEye = landmarks[263]; // Right eye outer
    const noseTip = landmarks[1];    // Nose tip
    
    // Canonical UV positions (must match face-texture-extractor.js)
    const dstLeftEye = { x: 0.30, y: 0.35 };
    const dstRightEye = { x: 0.70, y: 0.35 };
    const dstNose = { x: 0.50, y: 0.55 };
    
    // Compute affine transform from live face landmarks to canonical UV space
    const transform = this.computeAffineTransform(
      { x: leftEye.x, y: leftEye.y },
      { x: rightEye.x, y: rightEye.y },
      { x: noseTip.x, y: noseTip.y },
      dstLeftEye, dstRightEye, dstNose
    );
    
    // Convert landmarks to vertex positions AND compute texture coords
    const positions = new Float32Array(numVertices * 2);
    const texCoords = new Float32Array(numVertices * 2);
    
    for (let i = 0; i < numVertices; i++) {
      // Screen positions
      positions[i * 2] = landmarks[i].x * canvasWidth;
      positions[i * 2 + 1] = landmarks[i].y * canvasHeight;
      
      // Texture coordinates: use affine transform to map to canonical UV space
      if (transform) {
        const srcX = landmarks[i].x;
        const srcY = landmarks[i].y;
        texCoords[i * 2] = transform.a * srcX + transform.b * srcY + transform.c;
        texCoords[i * 2 + 1] = transform.d * srcX + transform.e * srcY + transform.f;
      } else {
        // Fallback to basic mapping
        texCoords[i * 2] = landmarks[i].x;
        texCoords[i * 2 + 1] = landmarks[i].y;
      }
    }
    
    // Upload positions
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
    
    // Upload dynamic texture coordinates
    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.DYNAMIC_DRAW);
    
    // Use program
    gl.useProgram(this.program);
    
    // Set up position attribute
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(this.locations.position);
    gl.vertexAttribPointer(this.locations.position, 2, gl.FLOAT, false, 0, 0);
    
    // Set up tex coord attribute
    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(this.locations.texCoord);
    gl.vertexAttribPointer(this.locations.texCoord, 2, gl.FLOAT, false, 0, 0);
    
    // Set uniforms
    gl.uniform2f(this.locations.resolution, this.canvas.width, this.canvas.height);
    gl.uniform1f(this.locations.opacity, opacity);
    gl.uniform1f(this.locations.edgeFeather, edgeFeather);
    
    // Bind texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
    gl.uniform1i(this.locations.texture, 0);
    
    // Use face mesh triangulation
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(FACE_MESH_TRIANGULATION), gl.DYNAMIC_DRAW);
    
    // Draw all face triangles
    gl.drawElements(gl.TRIANGLES, FACE_MESH_TRIANGULATION.length, gl.UNSIGNED_SHORT, 0);
  }
  
  // Compute affine transform from 3 source points to 3 destination points
  computeAffineTransform(s0, s1, s2, d0, d1, d2) {
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
  
  // Load a face swap texture from a canvas
  loadFaceSwapTexture(canvas) {
    this.createTextureFromImage(canvas);
    console.log('Face swap texture loaded:', canvas.width, 'x', canvas.height);
  }
  
  // Draw entire face mesh for debugging
  drawFullMesh(landmarks, canvasWidth, canvasHeight, opacity = 0.5) {
    if (!this.maskLoaded || !landmarks || landmarks.length < 468) {
      return;
    }
    
    const gl = this.gl;
    
    // Only use first 468 landmarks (the base mesh)
    const numVertices = 468;
    
    // Convert landmarks to vertex positions
    const positions = new Float32Array(numVertices * 2);
    for (let i = 0; i < numVertices; i++) {
      positions[i * 2] = landmarks[i].x * canvasWidth;
      positions[i * 2 + 1] = landmarks[i].y * canvasHeight;
    }
    
    // Upload positions
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
    
    // Use program
    gl.useProgram(this.program);
    
    // Set up position attribute
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(this.locations.position);
    gl.vertexAttribPointer(this.locations.position, 2, gl.FLOAT, false, 0, 0);
    
    // Set up tex coord attribute
    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(this.locations.texCoord);
    gl.vertexAttribPointer(this.locations.texCoord, 2, gl.FLOAT, false, 0, 0);
    
    // Set uniforms
    gl.uniform2f(this.locations.resolution, canvasWidth, canvasHeight);
    gl.uniform1f(this.locations.opacity, opacity);
    
    // Bind texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
    gl.uniform1i(this.locations.texture, 0);
    
    // Use full triangulation
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(FACE_MESH_TRIANGULATION), gl.DYNAMIC_DRAW);
    
    // Draw all triangles
    gl.drawElements(gl.TRIANGLES, FACE_MESH_TRIANGULATION.length, gl.UNSIGNED_SHORT, 0);
  }
}
