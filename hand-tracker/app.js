// Face Filter App
// Uses BlazeFace for face detection + mesh overlay
// Uses MediaPipe Hands for hand tracking

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const countDisplay = document.getElementById('count');
const statusDisplay = document.getElementById('status');

let faceModel = null;
let handDetector = null;

// Hand tracking config
const FINGER_TIPS = [4, 8, 12, 16, 20];
const FINGER_PIPS = [3, 6, 10, 14, 18];
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
    [5, 9], [9, 13], [13, 17]
];

// Initialize webcam
async function setupCamera() {
    statusDisplay.textContent = 'Accessing camera...';
    
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
    });
    
    video.srcObject = stream;
    
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            video.play();
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            resolve();
        };
    });
}

// Load models
async function loadModels() {
    statusDisplay.textContent = 'Loading TensorFlow.js...';
    await tf.ready();
    console.log('TensorFlow.js ready, backend:', tf.getBackend());
    
    statusDisplay.textContent = 'Loading face detection...';
    faceModel = await blazeface.load();
    console.log('BlazeFace loaded');
    
    statusDisplay.textContent = 'Loading hand detection...';
    handDetector = await handPoseDetection.createDetector(
        handPoseDetection.SupportedModels.MediaPipeHands,
        {
            runtime: 'mediapipe',
            solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands',
            modelType: 'full',
            maxHands: 2
        }
    );
    console.log('Hand detector loaded');
}

// Count fingers
function countFingers(landmarks) {
    if (!landmarks || landmarks.length < 21) return 0;
    let count = 0;
    
    const wrist = landmarks[0];
    const thumbTip = landmarks[4];
    const thumbIP = landmarks[3];
    
    const thumbTipToWrist = Math.hypot(thumbTip.x - wrist.x, thumbTip.y - wrist.y);
    const thumbIPToWrist = Math.hypot(thumbIP.x - wrist.x, thumbIP.y - wrist.y);
    if (thumbTipToWrist > thumbIPToWrist * 1.1) count++;
    
    for (let i = 1; i < 5; i++) {
        const tip = landmarks[FINGER_TIPS[i]];
        const pip = landmarks[FINGER_PIPS[i]];
        if (tip.y < pip.y) count++;
    }
    
    return count;
}

// Draw face mesh
function drawFaceMesh(face) {
    // BlazeFace returns tensors or arrays - handle both
    let topLeft, bottomRight, landmarks;
    
    if (face.topLeft.arraySync) {
        topLeft = face.topLeft.arraySync();
        bottomRight = face.bottomRight.arraySync();
        landmarks = face.landmarks.arraySync();
    } else {
        topLeft = face.topLeft;
        bottomRight = face.bottomRight;
        landmarks = face.landmarks;
    }
    
    console.log('Drawing face at:', topLeft, bottomRight);
    
    // Use coordinates directly - CSS handles mirroring
    const centerX = (topLeft[0] + bottomRight[0]) / 2;
    const centerY = (topLeft[1] + bottomRight[1]) / 2;
    const width = bottomRight[0] - topLeft[0];
    const height = bottomRight[1] - topLeft[1];
    
    // Draw a rectangle around face
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
    ctx.lineWidth = 4;
    ctx.strokeRect(topLeft[0], topLeft[1], width, height);
    
    // Draw a BIG circle at face center
    ctx.beginPath();
    ctx.arc(centerX, centerY, 50, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 255, 0, 0.7)';
    ctx.fill();
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 5;
    ctx.stroke();
    
    const radiusX = width * 0.55;
    const radiusY = height * 0.7;
    const adjustedCenterY = centerY - height * 0.05;
    
    // Draw concentric oval mesh
    const rings = 8;
    const pointsPerRing = 24;
    const allPoints = [];
    
    allPoints.push({ x: centerX, y: adjustedCenterY });
    
    for (let ring = 1; ring <= rings; ring++) {
        const ringRatio = ring / rings;
        const rx = radiusX * ringRatio;
        const ry = radiusY * ringRatio;
        
        for (let i = 0; i < pointsPerRing; i++) {
            const angle = (i / pointsPerRing) * Math.PI * 2 - Math.PI / 2;
            allPoints.push({
                x: centerX + Math.cos(angle) * rx,
                y: adjustedCenterY + Math.sin(angle) * ry,
                ring, idx: i
            });
        }
    }
    
    // Draw ring connections
    ctx.strokeStyle = 'rgba(0, 217, 255, 0.6)';
    ctx.lineWidth = 0.8;
    
    for (let ring = 1; ring <= rings; ring++) {
        const startIdx = 1 + (ring - 1) * pointsPerRing;
        for (let i = 0; i < pointsPerRing; i++) {
            const p1 = allPoints[startIdx + i];
            const p2 = allPoints[startIdx + ((i + 1) % pointsPerRing)];
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
        }
    }
    
    // Radial lines from center
    for (let i = 0; i < pointsPerRing; i += 2) {
        ctx.beginPath();
        ctx.moveTo(allPoints[0].x, allPoints[0].y);
        ctx.lineTo(allPoints[1 + i].x, allPoints[1 + i].y);
        ctx.stroke();
    }
    
    // Connect between rings
    for (let ring = 1; ring < rings; ring++) {
        const innerStart = 1 + (ring - 1) * pointsPerRing;
        const outerStart = 1 + ring * pointsPerRing;
        
        for (let i = 0; i < pointsPerRing; i++) {
            ctx.beginPath();
            ctx.moveTo(allPoints[innerStart + i].x, allPoints[innerStart + i].y);
            ctx.lineTo(allPoints[outerStart + i].x, allPoints[outerStart + i].y);
            ctx.stroke();
        }
    }
    
    // Draw mesh points
    for (const point of allPoints) {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 255, 136, 0.7)';
        ctx.fill();
    }
    
    // Draw facial landmarks
    if (landmarks && landmarks.length) {
        for (const lm of landmarks) {
            const x = Array.isArray(lm) ? lm[0] : lm.x || lm[0];
            const y = Array.isArray(lm) ? lm[1] : lm.y || lm[1];
            
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fillStyle = '#00ff88';
            ctx.fill();
            
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(0, 255, 136, 0.5)';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    }
}

// Draw hand
function drawHand(landmarks) {
    ctx.strokeStyle = 'rgba(0, 217, 255, 0.7)';
    ctx.lineWidth = 3;
    
    for (const [start, end] of HAND_CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(landmarks[start].x, landmarks[start].y);
        ctx.lineTo(landmarks[end].x, landmarks[end].y);
        ctx.stroke();
    }
    
    for (let i = 0; i < landmarks.length; i++) {
        const isTip = FINGER_TIPS.includes(i);
        ctx.beginPath();
        ctx.arc(landmarks[i].x, landmarks[i].y, isTip ? 8 : 5, 0, Math.PI * 2);
        ctx.fillStyle = isTip ? '#00ff88' : '#00d9ff';
        ctx.fill();
    }
}

// Main detection loop
async function detect() {
    if (!faceModel) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    try {
        // Detect faces
        const faces = await faceModel.estimateFaces(video, false);
        
        for (const face of faces) {
            drawFaceMesh(face);
        }
        
        // Detect hands
        let totalFingers = 0;
        if (handDetector) {
            const hands = await handDetector.estimateHands(video);
            for (const hand of hands) {
                drawHand(hand.keypoints);
                totalFingers += countFingers(hand.keypoints);
            }
        }
        
        countDisplay.textContent = totalFingers;
        
        // Status
        const parts = [];
        if (faces.length > 0) parts.push(`${faces.length} face${faces.length > 1 ? 's' : ''}`);
        if (handDetector) {
            // hands already processed above
        }
        statusDisplay.textContent = parts.length > 0 ? `Tracking ${parts.join(', ')}` : 'Show your face';
        
    } catch (error) {
        console.error('Detection error:', error);
    }
    
    requestAnimationFrame(detect);
}

// Initialize
async function init() {
    try {
        await setupCamera();
        await loadModels();
        
        statusDisplay.textContent = 'Ready!';
        statusDisplay.classList.add('ready');
        
        detect();
    } catch (error) {
        console.error('Init error:', error);
        statusDisplay.textContent = 'Error: ' + error.message;
        statusDisplay.classList.add('error');
    }
}

init();
