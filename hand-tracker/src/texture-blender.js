// Texture Blender - Multi-view face texture composition
// Simple averaging approach for reliable blending

export class TextureBlender {
  constructor(outputSize = 512) {
    this.outputSize = outputSize;
    this.textures = []; // Array of { canvas, weight, angle, landmarks }
  }

  // Add a texture with metadata
  addTexture(canvas, weight = 1.0, angle = 'front', landmarks = null) {
    console.log('TextureBlender: Adding texture with angle:', angle);
    this.textures.push({ canvas, weight, angle, landmarks });
  }

  // Clear all textures
  clear() {
    this.textures = [];
  }

  // Main blend function
  blend() {
    console.log('TextureBlender: Blending', this.textures.length, 'textures');
    
    if (this.textures.length === 0) return null;
    if (this.textures.length === 1) return this.copyCanvas(this.textures[0].canvas);
    
    // Use simple weighted averaging for reliable results
    return this.blendAverage();
  }

  // Simple averaging blend - combines all textures with equal weight
  blendAverage() {
    const size = this.outputSize;
    
    // Create output canvas
    const outputCanvas = document.createElement('canvas');
    outputCanvas.width = size;
    outputCanvas.height = size;
    const ctx = outputCanvas.getContext('2d');
    const outputData = ctx.createImageData(size, size);
    const output = outputData.data;
    
    // Get all texture data
    const textureData = this.textures.map(tex => {
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = size;
      tempCanvas.height = size;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.drawImage(tex.canvas, 0, 0, size, size);
      return {
        data: tempCtx.getImageData(0, 0, size, size).data,
        angle: tex.angle,
        weight: tex.weight
      };
    });
    
    console.log('TextureBlender: Processing', textureData.length, 'texture(s)');
    
    // For each pixel, average all textures that have content there
    for (let i = 0; i < output.length; i += 4) {
      let totalR = 0, totalG = 0, totalB = 0, totalA = 0;
      let totalWeight = 0;
      
      for (const tex of textureData) {
        const alpha = tex.data[i + 3];
        
        // Only include pixels with significant alpha
        if (alpha > 30) {
          const weight = (alpha / 255) * tex.weight;
          totalR += tex.data[i] * weight;
          totalG += tex.data[i + 1] * weight;
          totalB += tex.data[i + 2] * weight;
          totalA += alpha * weight;
          totalWeight += weight;
        }
      }
      
      if (totalWeight > 0) {
        output[i] = totalR / totalWeight;
        output[i + 1] = totalG / totalWeight;
        output[i + 2] = totalB / totalWeight;
        output[i + 3] = Math.min(255, totalA / totalWeight);
      }
    }
    
    ctx.putImageData(outputData, 0, 0);
    
    console.log('TextureBlender: Blend complete');
    return outputCanvas;
  }

  // Pose-aware blending - uses front for center, sides for edges
  blendPoseAware() {
    const size = this.outputSize;
    
    // Categorize textures by pose
    const frontTextures = this.textures.filter(t => t.angle === 'front');
    const leftTextures = this.textures.filter(t => t.angle === 'left');
    const rightTextures = this.textures.filter(t => t.angle === 'right');
    
    console.log('Pose-aware blend - front:', frontTextures.length,
                'left:', leftTextures.length, 'right:', rightTextures.length);
    
    // If we have no pose variety, use simple averaging
    if (frontTextures.length === this.textures.length) {
      return this.blendAverage();
    }
    
    // Create output
    const outputCanvas = document.createElement('canvas');
    outputCanvas.width = size;
    outputCanvas.height = size;
    const ctx = outputCanvas.getContext('2d');
    const outputData = ctx.createImageData(size, size);
    const output = outputData.data;
    
    // Get averaged data for each pose group
    const getGroupAverage = (textures) => {
      if (textures.length === 0) return null;
      
      const avgCanvas = document.createElement('canvas');
      avgCanvas.width = size;
      avgCanvas.height = size;
      const avgCtx = avgCanvas.getContext('2d');
      const avgData = avgCtx.createImageData(size, size);
      
      const allData = textures.map(t => {
        const c = document.createElement('canvas');
        c.width = size; c.height = size;
        const cCtx = c.getContext('2d');
        cCtx.drawImage(t.canvas, 0, 0, size, size);
        return cCtx.getImageData(0, 0, size, size).data;
      });
      
      for (let i = 0; i < avgData.data.length; i += 4) {
        let r = 0, g = 0, b = 0, a = 0, count = 0;
        for (const d of allData) {
          if (d[i + 3] > 30) {
            r += d[i];
            g += d[i + 1];
            b += d[i + 2];
            a += d[i + 3];
            count++;
          }
        }
        if (count > 0) {
          avgData.data[i] = r / count;
          avgData.data[i + 1] = g / count;
          avgData.data[i + 2] = b / count;
          avgData.data[i + 3] = a / count;
        }
      }
      
      avgCtx.putImageData(avgData, 0, 0);
      return avgCtx.getImageData(0, 0, size, size);
    };
    
    const frontData = getGroupAverage(frontTextures.length > 0 ? frontTextures : this.textures);
    const leftData = getGroupAverage(leftTextures);
    const rightData = getGroupAverage(rightTextures);
    
    // Blend based on X position
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = (y * size + x) * 4;
        const nx = x / size; // 0 to 1
        
        // Calculate weights based on position
        // Left side (0-0.3): prefer left-pose textures
        // Center (0.3-0.7): prefer front textures
        // Right side (0.7-1): prefer right-pose textures
        
        let leftW = 0, frontW = 1, rightW = 0;
        
        if (nx < 0.35) {
          leftW = 1 - (nx / 0.35);
          frontW = nx / 0.35;
        } else if (nx > 0.65) {
          rightW = (nx - 0.65) / 0.35;
          frontW = 1 - rightW;
        }
        
        let r = 0, g = 0, b = 0, a = 0, totalW = 0;
        
        // Add front contribution
        if (frontData && frontData.data[i + 3] > 30) {
          r += frontData.data[i] * frontW;
          g += frontData.data[i + 1] * frontW;
          b += frontData.data[i + 2] * frontW;
          a += frontData.data[i + 3] * frontW;
          totalW += frontW;
        }
        
        // Add left contribution (left-pose shows right side of face)
        if (leftData && leftData.data[i + 3] > 30 && rightW > 0) {
          r += leftData.data[i] * rightW;
          g += leftData.data[i + 1] * rightW;
          b += leftData.data[i + 2] * rightW;
          a += leftData.data[i + 3] * rightW;
          totalW += rightW;
        }
        
        // Add right contribution (right-pose shows left side of face)
        if (rightData && rightData.data[i + 3] > 30 && leftW > 0) {
          r += rightData.data[i] * leftW;
          g += rightData.data[i + 1] * leftW;
          b += rightData.data[i + 2] * leftW;
          a += rightData.data[i + 3] * leftW;
          totalW += leftW;
        }
        
        if (totalW > 0) {
          output[i] = r / totalW;
          output[i + 1] = g / totalW;
          output[i + 2] = b / totalW;
          output[i + 3] = a / totalW;
        }
      }
    }
    
    ctx.putImageData(outputData, 0, 0);
    return outputCanvas;
  }

  // Helper to copy a canvas
  copyCanvas(sourceCanvas) {
    const copy = document.createElement('canvas');
    copy.width = sourceCanvas.width;
    copy.height = sourceCanvas.height;
    const ctx = copy.getContext('2d');
    ctx.drawImage(sourceCanvas, 0, 0);
    return copy;
  }
}
