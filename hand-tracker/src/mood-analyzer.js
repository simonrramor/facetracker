// Mood Analyzer - Combines facial expressions to infer emotional state
// Uses FACS (Facial Action Coding System) mappings and Russell's Circumplex Model
// Reference: Russell, J. A. (1980). A circumplex model of affect.

/**
 * Emotion detection based on FACS Action Unit to emotion mappings
 * Maps MediaPipe blendshapes to the 6 basic emotions
 */
const EMOTION_MAPPINGS = {
  // Happy: AU6 (Cheek Raiser) + AU12 (Lip Corner Puller)
  happy: {
    blendshapes: ['mouthSmileLeft', 'mouthSmileRight', 'cheekSquintLeft', 'cheekSquintRight'],
    weights: [0.35, 0.35, 0.15, 0.15],
    threshold: 0.3
  },
  
  // Sad: AU1 (Inner Brow Raiser) + AU4 (Brow Lowerer) + AU15 (Lip Corner Depressor)
  sad: {
    blendshapes: ['browInnerUp', 'browDownLeft', 'browDownRight', 'mouthFrownLeft', 'mouthFrownRight'],
    weights: [0.2, 0.15, 0.15, 0.25, 0.25],
    threshold: 0.25
  },
  
  // Angry: AU4 (Brow Lowerer) + AU5 (Upper Lid Raiser) + AU7 (Lid Tightener) + AU23 (Lip Tightener)
  angry: {
    blendshapes: ['browDownLeft', 'browDownRight', 'eyeSquintLeft', 'eyeSquintRight', 
                  'mouthPressLeft', 'mouthPressRight', 'noseSneerLeft', 'noseSneerRight'],
    weights: [0.15, 0.15, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1],
    threshold: 0.3
  },
  
  // Fear: AU1+2 (Brow Raise) + AU5 (Upper Lid Raiser) + AU20 (Lip Stretcher) + AU26 (Jaw Drop)
  fear: {
    blendshapes: ['browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 
                  'eyeWideLeft', 'eyeWideRight', 'mouthStretchLeft', 'mouthStretchRight', 'jawOpen'],
    weights: [0.15, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.15],
    threshold: 0.35
  },
  
  // Surprise: AU1+2 (Brow Raise) + AU5 (Upper Lid Raiser) + AU26 (Jaw Drop)
  surprise: {
    blendshapes: ['browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 
                  'eyeWideLeft', 'eyeWideRight', 'jawOpen'],
    weights: [0.2, 0.1, 0.1, 0.2, 0.2, 0.2],
    threshold: 0.35
  },
  
  // Disgust: AU9 (Nose Wrinkler) + AU15 (Lip Corner Depressor)
  disgust: {
    blendshapes: ['noseSneerLeft', 'noseSneerRight', 'mouthFrownLeft', 'mouthFrownRight', 
                  'browDownLeft', 'browDownRight'],
    weights: [0.25, 0.25, 0.15, 0.15, 0.1, 0.1],
    threshold: 0.3
  }
};

/**
 * Mood descriptors based on valence-arousal quadrants
 * Using Russell's Circumplex Model of Affect
 */
const MOOD_QUADRANTS = {
  // Q1: High Arousal, Positive Valence
  excited: { valence: [0.3, 1], arousal: [0.5, 1], label: 'Excited' },
  happy: { valence: [0.2, 1], arousal: [0, 0.5], label: 'Happy' },
  elated: { valence: [0.5, 1], arousal: [0.3, 0.7], label: 'Elated' },
  
  // Q2: High Arousal, Negative Valence
  angry: { valence: [-1, -0.3], arousal: [0.4, 1], label: 'Angry' },
  stressed: { valence: [-0.5, 0], arousal: [0.3, 0.7], label: 'Stressed' },
  tense: { valence: [-0.4, 0], arousal: [0.2, 0.5], label: 'Tense' },
  
  // Q3: Low Arousal, Negative Valence
  sad: { valence: [-1, -0.2], arousal: [-1, 0], label: 'Sad' },
  bored: { valence: [-0.3, 0.1], arousal: [-0.7, -0.2], label: 'Bored' },
  tired: { valence: [-0.2, 0.2], arousal: [-1, -0.5], label: 'Tired' },
  
  // Q4: Low Arousal, Positive Valence
  relaxed: { valence: [0.2, 0.6], arousal: [-0.7, -0.2], label: 'Relaxed' },
  calm: { valence: [0, 0.4], arousal: [-0.5, 0], label: 'Calm' },
  content: { valence: [0.1, 0.5], arousal: [-0.3, 0.2], label: 'Content' },
  
  // Center: Neutral
  neutral: { valence: [-0.15, 0.15], arousal: [-0.15, 0.15], label: 'Neutral' }
};

/**
 * MoodAnalyzer class - analyzes facial expressions to determine mood
 */
export class MoodAnalyzer {
  constructor() {
    // Smoothing buffer for temporal averaging
    this.historySize = 10;
    this.emotionHistory = [];
    this.valenceHistory = [];
    this.arousalHistory = [];
    
    // Current state
    this.currentEmotions = {};
    this.currentValence = 0;
    this.currentArousal = 0;
    this.currentMood = 'Neutral';
    this.dominantEmotion = 'neutral';
    
    // Settings
    this.smoothingFactor = 0.3; // 0 = no smoothing, 1 = max smoothing
  }

  /**
   * Set the smoothing factor for temporal averaging
   * @param {number} factor - 0 to 1
   */
  setSmoothingFactor(factor) {
    this.smoothingFactor = Math.max(0, Math.min(1, factor));
    this.historySize = Math.round(5 + factor * 15); // 5-20 frames
  }

  /**
   * Calculate emotion scores from raw blendshape data
   * @param {Object} rawScores - Object with blendshape names as keys and scores (0-1) as values
   * @returns {Object} Emotion scores for each basic emotion
   */
  calculateEmotions(rawScores) {
    if (!rawScores) return {};

    const emotions = {};

    for (const [emotion, mapping] of Object.entries(EMOTION_MAPPINGS)) {
      let score = 0;
      
      for (let i = 0; i < mapping.blendshapes.length; i++) {
        const blendshape = mapping.blendshapes[i];
        const weight = mapping.weights[i];
        const value = rawScores[blendshape] || 0;
        score += value * weight;
      }
      
      // Normalize and apply threshold
      emotions[emotion] = Math.max(0, score);
    }

    return emotions;
  }

  /**
   * Calculate valence (positive/negative) from emotion scores
   * Valence = (Happy + Surprise) - (Sad + Angry + Fear + Disgust)
   * @param {Object} emotions - Emotion scores
   * @returns {number} Valence value (-1 to 1)
   */
  calculateValence(emotions) {
    const positive = (emotions.happy || 0) * 1.2 + (emotions.surprise || 0) * 0.3;
    const negative = (emotions.sad || 0) * 0.8 + 
                     (emotions.angry || 0) * 1.0 + 
                     (emotions.fear || 0) * 0.6 + 
                     (emotions.disgust || 0) * 0.8;
    
    let valence = positive - negative;
    
    // Normalize to -1 to 1 range
    valence = Math.max(-1, Math.min(1, valence * 2));
    
    return valence;
  }

  /**
   * Calculate arousal (activation level) from emotion scores
   * Arousal = (Happy + Angry + Fear + Surprise) - (Sad + Neutral)
   * @param {Object} emotions - Emotion scores
   * @returns {number} Arousal value (-1 to 1)
   */
  calculateArousal(emotions) {
    const high = (emotions.happy || 0) * 0.6 + 
                 (emotions.angry || 0) * 1.0 + 
                 (emotions.fear || 0) * 1.0 + 
                 (emotions.surprise || 0) * 0.8;
    const low = (emotions.sad || 0) * 0.8;
    
    let arousal = high - low;
    
    // Normalize to -1 to 1 range
    arousal = Math.max(-1, Math.min(1, arousal * 2));
    
    return arousal;
  }

  /**
   * Determine the dominant emotion from scores
   * @param {Object} emotions - Emotion scores
   * @returns {string} Name of dominant emotion
   */
  getDominantEmotion(emotions) {
    let maxScore = 0;
    let dominant = 'neutral';
    
    for (const [emotion, score] of Object.entries(emotions)) {
      const threshold = EMOTION_MAPPINGS[emotion]?.threshold || 0.25;
      if (score > maxScore && score > threshold) {
        maxScore = score;
        dominant = emotion;
      }
    }
    
    return dominant;
  }

  /**
   * Determine mood label from valence-arousal coordinates
   * @param {number} valence - Valence value (-1 to 1)
   * @param {number} arousal - Arousal value (-1 to 1)
   * @returns {string} Mood label
   */
  getMoodFromVA(valence, arousal) {
    // Check each mood quadrant
    for (const [moodKey, bounds] of Object.entries(MOOD_QUADRANTS)) {
      const [vMin, vMax] = bounds.valence;
      const [aMin, aMax] = bounds.arousal;
      
      if (valence >= vMin && valence <= vMax && arousal >= aMin && arousal <= aMax) {
        return bounds.label;
      }
    }
    
    // Fallback based on quadrant
    if (valence > 0 && arousal > 0) return 'Happy';
    if (valence < 0 && arousal > 0) return 'Stressed';
    if (valence < 0 && arousal < 0) return 'Sad';
    if (valence > 0 && arousal < 0) return 'Calm';
    
    return 'Neutral';
  }

  /**
   * Apply temporal smoothing to a value
   * @param {number[]} history - History buffer
   * @param {number} newValue - New value to add
   * @param {number} maxSize - Maximum history size
   * @returns {number} Smoothed value
   */
  smoothValue(history, newValue, maxSize) {
    history.push(newValue);
    if (history.length > maxSize) {
      history.shift();
    }
    
    // Weighted average (more recent values have more weight)
    let sum = 0;
    let weightSum = 0;
    for (let i = 0; i < history.length; i++) {
      const weight = (i + 1) / history.length;
      sum += history[i] * weight;
      weightSum += weight;
    }
    
    return sum / weightSum;
  }

  /**
   * Main analysis function - process blendshape data and return mood analysis
   * @param {Object} rawScores - Raw blendshape scores from MediaPipe
   * @returns {Object} Complete mood analysis result
   */
  analyze(rawScores) {
    if (!rawScores) {
      return {
        emotions: {},
        dominantEmotion: 'neutral',
        valence: 0,
        arousal: 0,
        mood: 'Neutral',
        confidence: 0
      };
    }

    // Calculate base emotions
    const emotions = this.calculateEmotions(rawScores);
    
    // Calculate valence and arousal
    let valence = this.calculateValence(emotions);
    let arousal = this.calculateArousal(emotions);
    
    // Apply smoothing if enabled
    if (this.smoothingFactor > 0) {
      const historySize = Math.round(5 + this.smoothingFactor * 15);
      valence = this.smoothValue(this.valenceHistory, valence, historySize);
      arousal = this.smoothValue(this.arousalHistory, arousal, historySize);
    }
    
    // Get dominant emotion and mood
    const dominantEmotion = this.getDominantEmotion(emotions);
    const mood = this.getMoodFromVA(valence, arousal);
    
    // Calculate confidence (based on emotion intensity)
    const maxEmotion = Math.max(...Object.values(emotions), 0);
    const confidence = Math.min(1, maxEmotion * 2);
    
    // Update current state
    this.currentEmotions = emotions;
    this.currentValence = valence;
    this.currentArousal = arousal;
    this.currentMood = mood;
    this.dominantEmotion = dominantEmotion;
    
    return {
      emotions,
      dominantEmotion,
      valence,
      arousal,
      mood,
      confidence
    };
  }

  /**
   * Get the current mood state without recalculating
   * @returns {Object} Current mood state
   */
  getCurrentState() {
    return {
      emotions: this.currentEmotions,
      dominantEmotion: this.dominantEmotion,
      valence: this.currentValence,
      arousal: this.currentArousal,
      mood: this.currentMood
    };
  }

  /**
   * Reset the analyzer state
   */
  reset() {
    this.emotionHistory = [];
    this.valenceHistory = [];
    this.arousalHistory = [];
    this.currentEmotions = {};
    this.currentValence = 0;
    this.currentArousal = 0;
    this.currentMood = 'Neutral';
    this.dominantEmotion = 'neutral';
  }
}

// Export a singleton instance
export const moodAnalyzer = new MoodAnalyzer();

// Export emotion mappings for reference
export { EMOTION_MAPPINGS, MOOD_QUADRANTS };
