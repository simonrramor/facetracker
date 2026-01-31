// Adaptive LOD (Level of Detail) Triangulation System
// Inspired by Wood et al. (2022) "3D face reconstruction with dense landmarks"
// https://arxiv.org/abs/2204.02776
// Synthesizes additional landmarks and provides higher triangle density
// around eyes, mouth, and nose with subdivision for flat regions

import Delaunator from 'delaunator';

// MediaPipe Face Mesh landmark indices organized by facial region
// These are the canonical 468 landmark indices
export const FACIAL_REGIONS = {
  // High detail regions (critical for expressions and identity)
  LEFT_EYE: {
    landmarks: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    detail: 'high',
    priority: 1
  },
  RIGHT_EYE: {
    landmarks: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    detail: 'high',
    priority: 1
  },
  LEFT_EYEBROW: {
    landmarks: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    detail: 'high',
    priority: 2
  },
  RIGHT_EYEBROW: {
    landmarks: [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    detail: 'high',
    priority: 2
  },
  UPPER_LIP: {
    landmarks: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78],
    detail: 'high',
    priority: 1
  },
  LOWER_LIP: {
    landmarks: [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61],
    detail: 'high',
    priority: 1
  },
  NOSE: {
    landmarks: [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 19, 94, 122, 351, 114, 343, 
                129, 358, 209, 429, 45, 275, 44, 274, 440, 278, 294, 460, 344, 278, 
                115, 218, 219, 237, 457, 438, 439, 48, 64, 240, 99, 235, 75, 60, 59, 166],
    detail: 'high',
    priority: 1
  },
  
  // Medium detail regions
  LEFT_CHEEK: {
    landmarks: [116, 117, 118, 119, 100, 101, 36, 205, 187, 123, 50, 147, 177, 137, 227, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148],
    detail: 'medium',
    priority: 3
  },
  RIGHT_CHEEK: {
    landmarks: [345, 346, 347, 348, 329, 330, 266, 425, 411, 352, 280, 376, 401, 366, 447, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377],
    detail: 'medium',
    priority: 3
  },
  CHIN: {
    landmarks: [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377],
    detail: 'medium',
    priority: 3
  },
  
  // Low detail regions (flatter areas)
  FOREHEAD: {
    landmarks: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    detail: 'low',
    priority: 4
  },
  FACE_OVAL: {
    landmarks: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    detail: 'low',
    priority: 5
  }
};

// Bilateral symmetry pairs - left landmark index maps to right landmark index
// This ensures the mesh is perfectly symmetric
export const SYMMETRY_PAIRS = {
  // Eyes
  33: 263, 7: 249, 163: 390, 144: 373, 145: 374, 153: 380, 154: 381, 155: 382,
  133: 362, 173: 398, 157: 384, 158: 385, 159: 386, 160: 387, 161: 388, 246: 466,
  // Eyebrows
  70: 300, 63: 293, 105: 334, 66: 296, 107: 336, 55: 285, 65: 295, 52: 282, 53: 283, 46: 276,
  // Nose (mostly center, but some pairs)
  129: 358, 209: 429, 45: 275, 44: 274,
  // Cheeks
  116: 345, 117: 346, 118: 347, 119: 348, 100: 329, 101: 330,
  36: 266, 205: 425, 187: 411, 123: 352, 50: 280, 147: 376, 177: 401,
  137: 366, 227: 447, 234: 454, 93: 323, 132: 361, 58: 288, 172: 397,
  136: 365, 150: 379, 149: 378, 176: 400, 148: 377,
  // Face oval
  127: 356, 162: 389, 21: 251, 54: 284, 103: 332, 67: 297, 109: 338
};

// Edges to interpolate for dense landmark synthesis
// These create additional points between existing landmarks in sparse regions
// Format: [landmark1, landmark2, numPoints] - creates numPoints between them
const DENSE_INTERPOLATION_EDGES = [
  // Left cheek - horizontal spans
  [116, 123, 2], [123, 147, 2], [147, 177, 2],
  [117, 50, 2], [50, 187, 2], 
  [118, 101, 2], [101, 205, 2],
  [119, 100, 2], [100, 36, 2],
  // Right cheek - horizontal spans  
  [345, 352, 2], [352, 376, 2], [376, 401, 2],
  [346, 280, 2], [280, 411, 2],
  [347, 330, 2], [330, 425, 2],
  [348, 329, 2], [329, 266, 2],
  // Left cheek - vertical connections
  [116, 117, 1], [117, 118, 1], [118, 119, 1],
  [123, 50, 1], [50, 101, 1], [101, 100, 1],
  [147, 187, 1], [187, 205, 1], [205, 36, 1],
  // Right cheek - vertical connections
  [345, 346, 1], [346, 347, 1], [347, 348, 1],
  [352, 280, 1], [280, 330, 1], [330, 329, 1],
  [376, 411, 1], [411, 425, 1], [425, 266, 1],
  // Forehead - horizontal
  [109, 10, 2], [10, 338, 2],
  [67, 103, 1], [103, 54, 1], [54, 21, 1],
  [297, 332, 1], [332, 284, 1], [284, 251, 1],
  // Forehead - connect across
  [109, 67, 1], [67, 103, 1],
  [338, 297, 1], [297, 332, 1],
  // Chin area
  [152, 148, 1], [148, 176, 1], [176, 149, 1],
  [152, 377, 1], [377, 400, 1], [400, 378, 1],
  // Jaw line densification
  [127, 162, 1], [162, 21, 1],
  [356, 389, 1], [389, 251, 1]
];

// Pre-computed adaptive triangulation that uses all 468 landmarks
// with variable density based on facial regions
// Implements dense landmark synthesis inspired by Wood et al. (2022)
export class AdaptiveTriangulation {
  constructor() {
    this.cachedTriangles = null;
    this.lastLandmarkHash = null;
    this.lodLevel = 1.0; // 0.0 = lowest, 1.0 = highest
    this.enableDenseLandmarks = true; // Synthesize additional landmarks
    this.enableSubdivision = true;    // Subdivide flat regions
    this.settings = {
      enableSymmetry: true,
      enableAdaptiveLOD: true,
      highDetailMultiplier: 1.0,
      mediumDetailMultiplier: 0.7,
      lowDetailMultiplier: 0.4,
      minTriangleArea: 0.00001,
      maxTriangleArea: 0.01
    };
  }

  // Update LOD settings
  setLOD(level) {
    this.lodLevel = Math.max(0, Math.min(1, level));
    this.cachedTriangles = null; // Invalidate cache
  }

  // Update settings
  updateSettings(newSettings) {
    this.settings = { ...this.settings, ...newSettings };
    this.cachedTriangles = null;
  }

  // Classify which region each landmark belongs to (for filtering)
  classifyLandmark(index) {
    for (const [regionName, region] of Object.entries(FACIAL_REGIONS)) {
      if (region.landmarks.includes(index)) {
        return { region: regionName, detail: region.detail, priority: region.priority };
      }
    }
    return { region: 'OTHER', detail: 'medium', priority: 3 };
  }

  // Get detail multiplier for a triangle based on its vertices' regions
  getTriangleDetailLevel(i0, i1, i2) {
    const c0 = this.classifyLandmark(i0);
    const c1 = this.classifyLandmark(i1);
    const c2 = this.classifyLandmark(i2);
    
    // Use the highest priority (lowest number = highest detail)
    const minPriority = Math.min(c0.priority, c1.priority, c2.priority);
    
    if (minPriority <= 1) return this.settings.highDetailMultiplier;
    if (minPriority <= 3) return this.settings.mediumDetailMultiplier;
    return this.settings.lowDetailMultiplier;
  }

  // Synthesize dense landmarks by interpolating between existing ones
  // This creates 700+ points from the original 468, similar to Wood et al. approach
  synthesizeDenseLandmarks(landmarks) {
    if (!this.enableDenseLandmarks || !landmarks || landmarks.length < 468) {
      return { landmarks, originalCount: landmarks ? landmarks.length : 0 };
    }

    const denseLandmarks = [...landmarks];
    const synthesizedIndices = new Map(); // Track which indices are synthesized
    
    for (const [idx1, idx2, numPoints] of DENSE_INTERPOLATION_EDGES) {
      if (idx1 >= landmarks.length || idx2 >= landmarks.length) continue;
      
      const p1 = landmarks[idx1];
      const p2 = landmarks[idx2];
      
      // Create interpolated points
      for (let i = 1; i <= numPoints; i++) {
        const t = i / (numPoints + 1);
        const newPoint = {
          x: p1.x + (p2.x - p1.x) * t,
          y: p1.y + (p2.y - p1.y) * t,
          z: (p1.z || 0) + ((p2.z || 0) - (p1.z || 0)) * t
        };
        
        const newIdx = denseLandmarks.length;
        denseLandmarks.push(newPoint);
        synthesizedIndices.set(newIdx, { from: idx1, to: idx2, t });
      }
    }
    
    // Add centroid points in large cheek regions for better coverage
    const cheekCentroids = [
      // Left cheek centroids
      [116, 123, 117, 50],
      [123, 147, 50, 187],
      [117, 50, 118, 101],
      [50, 187, 101, 205],
      // Right cheek centroids
      [345, 352, 346, 280],
      [352, 376, 280, 411],
      [346, 280, 347, 330],
      [280, 411, 330, 425]
    ];
    
    for (const quad of cheekCentroids) {
      const [i0, i1, i2, i3] = quad;
      if (i0 >= landmarks.length || i1 >= landmarks.length || 
          i2 >= landmarks.length || i3 >= landmarks.length) continue;
      
      const p0 = landmarks[i0];
      const p1 = landmarks[i1];
      const p2 = landmarks[i2];
      const p3 = landmarks[i3];
      
      // Add centroid
      const centroid = {
        x: (p0.x + p1.x + p2.x + p3.x) / 4,
        y: (p0.y + p1.y + p2.y + p3.y) / 4,
        z: ((p0.z || 0) + (p1.z || 0) + (p2.z || 0) + (p3.z || 0)) / 4
      };
      
      denseLandmarks.push(centroid);
    }
    
    console.log(`Dense landmarks: ${landmarks.length} -> ${denseLandmarks.length} points`);
    
    return { 
      landmarks: denseLandmarks, 
      originalCount: landmarks.length,
      synthesizedCount: denseLandmarks.length - landmarks.length
    };
  }

  // Subdivide triangles in flat/sparse regions (cheeks, forehead)
  subdivideTriangles(triangles, landmarks) {
    if (!this.enableSubdivision) return { triangles, landmarks };
    
    const newLandmarks = [...landmarks];
    const newTriangles = [];
    
    for (const [i0, i1, i2] of triangles) {
      const p0 = landmarks[i0];
      const p1 = landmarks[i1];
      const p2 = landmarks[i2];
      
      // Check if this triangle is in a low-detail region (cheeks/forehead)
      const region = this.classifyLandmark(i0);
      const isLowDetailRegion = region.detail === 'low' || region.detail === 'medium';
      
      // Calculate triangle area
      const area = this.triangleArea(p0, p1, p2);
      
      // Only subdivide large triangles in low-detail regions
      // Threshold: if area is larger than 2x median for the face
      const areaThreshold = 0.002; // ~0.2% of normalized face area
      
      if (isLowDetailRegion && area > areaThreshold && this.lodLevel > 0.5) {
        // Subdivide by adding centroid
        const centroid = {
          x: (p0.x + p1.x + p2.x) / 3,
          y: (p0.y + p1.y + p2.y) / 3,
          z: ((p0.z || 0) + (p1.z || 0) + (p2.z || 0)) / 3
        };
        
        const centroidIdx = newLandmarks.length;
        newLandmarks.push(centroid);
        
        // Create 3 triangles from 1
        newTriangles.push([i0, i1, centroidIdx]);
        newTriangles.push([i1, i2, centroidIdx]);
        newTriangles.push([i2, i0, centroidIdx]);
      } else {
        // Keep original triangle
        newTriangles.push([i0, i1, i2]);
      }
    }
    
    return { triangles: newTriangles, landmarks: newLandmarks };
  }

  // Generate Delaunay triangulation from landmarks
  generateDelaunayTriangles(landmarks) {
    if (!landmarks || landmarks.length < 3) return [];

    // Convert landmarks to flat coordinate array for Delaunator
    const coords = [];
    for (let i = 0; i < landmarks.length; i++) {
      coords.push(landmarks[i].x, landmarks[i].y);
    }

    try {
      const delaunay = new Delaunator(coords);
      const triangles = [];

      for (let i = 0; i < delaunay.triangles.length; i += 3) {
        const i0 = delaunay.triangles[i];
        const i1 = delaunay.triangles[i + 1];
        const i2 = delaunay.triangles[i + 2];

        triangles.push([i0, i1, i2]);
      }

      return triangles;
    } catch (e) {
      console.error('Delaunay triangulation failed:', e);
      return [];
    }
  }

  // Enforce bilateral symmetry by averaging symmetric landmark pairs
  enforceSymmetry(landmarks) {
    if (!this.settings.enableSymmetry) return landmarks;

    const symmetricLandmarks = [...landmarks];
    
    for (const [leftIdx, rightIdx] of Object.entries(SYMMETRY_PAIRS)) {
      const left = parseInt(leftIdx);
      const right = rightIdx;
      
      if (left < landmarks.length && right < landmarks.length) {
        const leftLm = landmarks[left];
        const rightLm = landmarks[right];
        
        // Average the Y and Z coordinates, mirror X around center (0.5)
        const avgY = (leftLm.y + rightLm.y) / 2;
        const avgZ = (leftLm.z + rightLm.z) / 2;
        
        // Calculate symmetric X positions
        const leftX = leftLm.x;
        const rightX = rightLm.x;
        const centerX = 0.5;
        const avgDistFromCenter = (Math.abs(leftX - centerX) + Math.abs(rightX - centerX)) / 2;
        
        symmetricLandmarks[left] = {
          x: centerX - avgDistFromCenter,
          y: avgY,
          z: avgZ
        };
        symmetricLandmarks[right] = {
          x: centerX + avgDistFromCenter,
          y: avgY,
          z: avgZ
        };
      }
    }

    return symmetricLandmarks;
  }

  // Calculate triangle area
  triangleArea(p0, p1, p2) {
    return Math.abs((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)) / 2;
  }

  // Calculate edge length between two points
  edgeLength(p0, p1) {
    const dx = p1.x - p0.x;
    const dy = p1.y - p0.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  // Check if triangle has reasonable edge lengths (not spanning across face)
  hasReasonableEdges(p0, p1, p2, maxEdgeLength) {
    const e0 = this.edgeLength(p0, p1);
    const e1 = this.edgeLength(p1, p2);
    const e2 = this.edgeLength(p2, p0);
    return e0 <= maxEdgeLength && e1 <= maxEdgeLength && e2 <= maxEdgeLength;
  }

  // Face oval landmark indices (defines the face boundary)
  // Complete clockwise ordering for proper polygon containment check
  getFaceOvalIndices() {
    return [
      // Start at forehead center, go clockwise
      10,   // top center
      338, 297, 332, 284, 251, 389, 356, // right side of forehead
      454, 323, 361, 288, 397, 365, 379, 378, 400, 377, // right jaw
      152, // chin center
      148, 176, 149, 150, 136, 172, 58, 132, 93, 234, // left jaw  
      127, 162, 21, 54, 103, 67, 109 // left side of forehead back to top
    ];
  }

  // Get a tighter inner boundary (for filtering edge triangles)
  getInnerBoundaryIndices() {
    return [
      // Tighter boundary that excludes jaw edges
      151, // forehead
      337, 299, 333, 298, 301, 368, 264, // right inner
      447, 366, 401, 435, 288, 361, 323, // right cheek to jaw
      152, 377, 400, 378, 379, // chin
      365, 397, 288, 172, 136, 150, 149, 176, 148, // left jaw
      383, 372, 345, 340, 261, // left cheek
      35, 124, 46, 53, 52, 65 // left inner to forehead
    ];
  }

  // Build face boundary polygon from landmarks
  buildFaceBoundary(landmarks) {
    const ovalIndices = this.getFaceOvalIndices();
    const boundary = [];
    
    for (const idx of ovalIndices) {
      if (idx < landmarks.length) {
        boundary.push({ x: landmarks[idx].x, y: landmarks[idx].y });
      }
    }
    
    return boundary;
  }

  // Check if a point is inside a polygon using ray casting
  pointInPolygon(point, polygon) {
    if (polygon.length < 3) return true; // Not enough points for a polygon
    
    let inside = false;
    const x = point.x;
    const y = point.y;
    
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i].x;
      const yi = polygon[i].y;
      const xj = polygon[j].x;
      const yj = polygon[j].y;
      
      if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
        inside = !inside;
      }
    }
    
    return inside;
  }

  // Check if triangle is inside face boundary
  // Requires ALL THREE vertices to be inside (stricter check)
  isTriangleInsideFace(p0, p1, p2, faceBoundary) {
    // Check all three vertices are inside
    const v0Inside = this.pointInPolygon(p0, faceBoundary);
    const v1Inside = this.pointInPolygon(p1, faceBoundary);
    const v2Inside = this.pointInPolygon(p2, faceBoundary);
    
    // All vertices must be inside
    if (!v0Inside || !v1Inside || !v2Inside) {
      return false;
    }
    
    // Also check centroid for extra safety
    const centroid = {
      x: (p0.x + p1.x + p2.x) / 3,
      y: (p0.y + p1.y + p2.y) / 3
    };
    
    return this.pointInPolygon(centroid, faceBoundary);
  }

  // Expand face boundary slightly to include edge landmarks
  expandBoundary(boundary, amount = 0.02) {
    if (boundary.length < 3) return boundary;
    
    // Find center
    let cx = 0, cy = 0;
    for (const p of boundary) {
      cx += p.x;
      cy += p.y;
    }
    cx /= boundary.length;
    cy /= boundary.length;
    
    // Expand each point outward from center
    return boundary.map(p => ({
      x: cx + (p.x - cx) * (1 + amount),
      y: cy + (p.y - cy) * (1 + amount)
    }));
  }

  // Filter triangles based on LOD and region
  // originalLandmarks: the original 468 landmarks (for boundary check)
  // landmarks: potentially dense landmarks (for geometry)
  filterTrianglesByLOD(triangles, landmarks, originalLandmarks = null) {
    const filtered = [];
    
    // Build face boundary from ORIGINAL landmarks (not synthesized)
    // This ensures the boundary matches the actual face outline
    const boundarySource = originalLandmarks || landmarks;
    const faceBoundary = this.buildFaceBoundary(boundarySource);
    
    // Slightly expand boundary to include edge landmarks
    const expandedBoundary = this.expandBoundary(faceBoundary, 0.01);
    
    // Calculate all edge lengths to find a reasonable max edge threshold
    const allEdgeLengths = [];
    for (const [i0, i1, i2] of triangles) {
      const p0 = landmarks[i0];
      const p1 = landmarks[i1];
      const p2 = landmarks[i2];
      allEdgeLengths.push(this.edgeLength(p0, p1));
      allEdgeLengths.push(this.edgeLength(p1, p2));
      allEdgeLengths.push(this.edgeLength(p2, p0));
    }
    
    // Sort and find a good threshold - use 80th percentile for tighter filtering
    allEdgeLengths.sort((a, b) => a - b);
    const percentile80 = allEdgeLengths[Math.floor(allEdgeLengths.length * 0.80)];
    const maxEdgeLength = percentile80 * 1.2; // Stricter: only 20% margin

    for (let i = 0; i < triangles.length; i++) {
      const [i0, i1, i2] = triangles[i];
      const p0 = landmarks[i0];
      const p1 = landmarks[i1];
      const p2 = landmarks[i2];
      
      // Filter out triangles with edges that are too long
      if (!this.hasReasonableEdges(p0, p1, p2, maxEdgeLength)) {
        continue;
      }
      
      // Filter out triangles that extend outside the face boundary
      if (!this.isTriangleInsideFace(p0, p1, p2, expandedBoundary)) {
        continue;
      }
      
      // At lower LOD levels, also filter based on area and region priority
      if (this.lodLevel < 0.99) {
        const area = this.triangleArea(p0, p1, p2);
        const detailMultiplier = this.getTriangleDetailLevel(i0, i1, i2);
        
        // Calculate median area for reference
        const areas = triangles.map(([a, b, c]) => {
          return this.triangleArea(landmarks[a], landmarks[b], landmarks[c]);
        });
        areas.sort((a, b) => a - b);
        const medianArea = areas[Math.floor(areas.length / 2)];
        
        // At lower LOD, filter small triangles in low-detail regions
        const minArea = medianArea * 0.05 * (1 - this.lodLevel) * (1 - detailMultiplier);
        
        if (area < minArea) {
          continue;
        }
      }
      
      filtered.push(triangles[i]);
    }

    return filtered;
  }

  // Subdivide triangles in high-detail regions
  subdivideHighDetailTriangles(triangles, landmarks) {
    const subdivided = [];
    const newLandmarks = [...landmarks];

    for (const [i0, i1, i2] of triangles) {
      const detailLevel = this.getTriangleDetailLevel(i0, i1, i2);
      
      // Only subdivide high-detail triangles
      if (detailLevel >= this.settings.highDetailMultiplier && this.lodLevel > 0.7) {
        const p0 = landmarks[i0];
        const p1 = landmarks[i1];
        const p2 = landmarks[i2];
        
        // Create midpoint
        const midpoint = {
          x: (p0.x + p1.x + p2.x) / 3,
          y: (p0.y + p1.y + p2.y) / 3,
          z: (p0.z + p1.z + p2.z) / 3
        };
        
        const midIdx = newLandmarks.length;
        newLandmarks.push(midpoint);
        
        // Create 3 triangles from 1
        subdivided.push([i0, i1, midIdx]);
        subdivided.push([i1, i2, midIdx]);
        subdivided.push([i2, i0, midIdx]);
      } else {
        subdivided.push([i0, i1, i2]);
      }
    }

    return { triangles: subdivided, landmarks: newLandmarks };
  }

  // Main method: Generate adaptive triangulation
  generateAdaptiveTriangulation(landmarks, forceRecompute = false) {
    if (!landmarks || landmarks.length < 3) {
      return { triangles: [], landmarks };
    }

    // Check if we can use cached result
    const landmarkHash = this.computeLandmarkHash(landmarks);
    if (!forceRecompute && this.cachedTriangles && this.lastLandmarkHash === landmarkHash) {
      return this.cachedTriangles;
    }

    // Step 1: Optionally enforce symmetry on original landmarks
    let processedLandmarks = this.enforceSymmetry(landmarks);
    
    // Store original 468 landmarks for boundary checking
    const originalLandmarks = [...processedLandmarks];

    // Step 2: Synthesize dense landmarks (700+ points from 468)
    // Inspired by Wood et al. (2022) dense landmark approach
    const denseResult = this.synthesizeDenseLandmarks(processedLandmarks);
    processedLandmarks = denseResult.landmarks;

    // Step 3: Generate Delaunay triangulation on dense landmarks
    let triangles = this.generateDelaunayTriangles(processedLandmarks);

    // Step 4: Filter by LOD, region, and face boundary
    // Use ORIGINAL landmarks for boundary (ensures triangles stay within face)
    triangles = this.filterTrianglesByLOD(triangles, processedLandmarks, originalLandmarks);

    // Step 5: Subdivide large triangles in flat regions (cheeks, forehead)
    const subdivisionResult = this.subdivideTriangles(triangles, processedLandmarks);
    triangles = subdivisionResult.triangles;
    processedLandmarks = subdivisionResult.landmarks;

    const result = {
      triangles,
      landmarks: processedLandmarks,
      stats: {
        originalLandmarks: landmarks.length,
        denseLandmarks: denseResult.landmarks.length,
        synthesizedPoints: denseResult.synthesizedCount || 0,
        triangleCount: triangles.length,
        lodLevel: this.lodLevel
      }
    };

    // Cache result
    this.cachedTriangles = result;
    this.lastLandmarkHash = landmarkHash;

    return result;
  }

  // Compute a simple hash of landmarks to detect changes
  computeLandmarkHash(landmarks) {
    if (!landmarks || landmarks.length === 0) return '';
    
    // Sample a few landmarks for quick comparison
    const samples = [0, 33, 133, 263, 362, 1, 168, 6];
    let hash = '';
    
    for (const idx of samples) {
      if (idx < landmarks.length) {
        const lm = landmarks[idx];
        hash += `${lm.x.toFixed(3)}${lm.y.toFixed(3)}`;
      }
    }
    
    return hash;
  }

  // Get triangles formatted for rendering (flat array of indices)
  getTriangleIndices(landmarks) {
    const { triangles } = this.generateAdaptiveTriangulation(landmarks);
    const indices = [];
    
    for (const [i0, i1, i2] of triangles) {
      indices.push(i0, i1, i2);
    }
    
    return indices;
  }

  // Get region info for debugging/visualization
  getRegionInfo(landmarkIndex) {
    return this.classifyLandmark(landmarkIndex);
  }
}

// Export a singleton instance for easy use
export const adaptiveTriangulation = new AdaptiveTriangulation();

// Export the Delaunator-based triangles as an alternative to MediaPipe's
export function computeDelaunayTriangulation(landmarks) {
  return adaptiveTriangulation.generateAdaptiveTriangulation(landmarks);
}
