// Turns raw landmarks into meaning. Everything in here is pure maths on plain {x, y} points
// (normalised 0..1, y down), so it can be tested with made-up landmarks and never touches the camera.
//
// Rule from the spec: Quark reacts to *movement and gestures*, never to appearance. Nothing here looks at
// what a person looks like — only where points are and how fast they move.

const hypot = Math.hypot;

// ----------------------------------------------------------------------------- hands

// MediaPipe hand landmark indices
const WRIST = 0, THUMB_IP = 3, THUMB_TIP = 4, INDEX_MCP = 5, MIDDLE_MCP = 9, PINKY_MCP = 17;
const FINGERS = [
  { name: 'index', pip: 6, tip: 8 },
  { name: 'middle', pip: 10, tip: 12 },
  { name: 'ring', pip: 14, tip: 16 },
  { name: 'pinky', pip: 18, tip: 20 },
];

/** Distance with x stretched by the frame's aspect ratio, so a circle in the world is a circle here. */
const dist = (a, b, aspect = 1) => hypot((a.x - b.x) * aspect, a.y - b.y);

/**
 * Describe one hand: which fingers are out, and what shape that makes.
 * shape: 'paper' | 'rock' | 'scissors' | 'thumbsup' | 'point' | 'unknown'
 */
export function classifyHand(lm, aspect = 4 / 3) {
  const palm = dist(lm[WRIST], lm[MIDDLE_MCP], aspect) || 1e-6;
  const ext = FINGERS.map((f) => dist(lm[WRIST], lm[f.tip], aspect) > dist(lm[WRIST], lm[f.pip], aspect) * 1.1);
  const curled = FINGERS.map((f) => dist(lm[WRIST], lm[f.tip], aspect) < dist(lm[WRIST], lm[f.pip], aspect) * 1.02);
  const n = ext.filter(Boolean).length;
  const thumbOut = dist(lm[THUMB_TIP], lm[MIDDLE_MCP], aspect) > palm * 0.95 && dist(lm[THUMB_TIP], lm[PINKY_MCP], aspect) > dist(lm[THUMB_IP], lm[PINKY_MCP], aspect) * 1.05;
  const thumbUp = thumbOut && lm[THUMB_TIP].y < lm[THUMB_IP].y - palm * 0.12 && lm[THUMB_TIP].y < lm[WRIST].y - palm * 0.7;
  const upright = lm[WRIST].y > lm[MIDDLE_MCP].y + palm * 0.25 / aspect; // fingers point up in the image

  let shape = 'unknown';
  if (n === 0 && curled.every(Boolean) && thumbUp) shape = 'thumbsup';
  else if (n === 4) shape = 'paper';
  else if (n === 0 && curled.filter(Boolean).length >= 3) shape = 'rock';
  else if (n === 2 && ext[0] && ext[1]) shape = 'scissors';
  else if (n === 1 && ext[0]) shape = 'point';

  let cx = 0, cy = 0;
  for (const p of lm) { cx += p.x; cy += p.y; }
  return { shape, n, ext, thumbOut, thumbUp, upright, palm, x: cx / lm.length, y: cy / lm.length, wrist: lm[WRIST] };
}

/** Counts left-right reversals of a hand's x position over a short window: that's a wave. */
export class WaveDetector {
  constructor({ windowMs = 1500, minSwing = 0.035, minReversals = 2 } = {}) {
    this.windowMs = windowMs;
    this.minSwing = minSwing;
    this.minReversals = minReversals;
    this.pts = [];
  }
  reset() { this.pts.length = 0; }
  /** Feed the wrist x each frame. Returns true the moment a wave is recognised. */
  push(t, x) {
    this.pts.push({ t, x });
    while (this.pts.length && t - this.pts[0].t > this.windowMs) this.pts.shift();
    if (this.pts.length < 5) return false;
    // zigzag: follow the hand's extreme in the current direction; a retreat of at least minSwing is a turn
    const m = this.minSwing;
    let dir = 0, ext = this.pts[0].x, turns = 0;
    for (const p of this.pts) {
      if (dir === 0) {
        if (p.x - ext >= m) { dir = 1; ext = p.x; }
        else if (ext - p.x >= m) { dir = -1; ext = p.x; }
      } else if (dir === 1) {
        if (p.x > ext) ext = p.x;
        else if (ext - p.x >= m) { dir = -1; turns++; ext = p.x; }
      } else if (p.x < ext) ext = p.x;
      else if (p.x - ext >= m) { dir = 1; turns++; ext = p.x; }
    }
    if (turns >= this.minReversals) { this.pts.length = 0; return true; }
    return false;
  }
}

/** True once `ok` has been continuously true for holdMs. Fires once per hold. */
export class Hold {
  constructor(holdMs) { this.holdMs = holdMs; this.since = 0; this.fired = false; }
  reset() { this.since = 0; this.fired = false; }
  update(t, ok) {
    if (!ok) { this.since = 0; this.fired = false; return false; }
    if (!this.since) this.since = t;
    if (!this.fired && t - this.since >= this.holdMs) { this.fired = true; return true; }
    return false;
  }
}

// ----------------------------------------------------------------------------- face & body

// MediaPipe face mesh indices: outer corners of the eyes and the nose tip
const EYE_R = 33, EYE_L = 263, NOSE = 1;

/**
 * Follows the nearest face over time: where it is, how big (= how close), how fast it moves, and whether
 * it just did something worth reacting to (came closer, backed off, left, returned, jumped, danced, startled).
 * `phoneMoving` suppresses movement events while the phone itself is being waved around.
 */
export class FaceMotion {
  constructor() {
    this.present = false;
    this.everSeen = false;
    this.lastSeen = 0;
    this.away = false;
    this.cx = 0.5; this.cy = 0.5;
    this.size = 0;
    this.fast = 0; this.slow = 0;       // two smoothed sizes; their ratio says "closer" or "farther"
    this.prev = null;
    this.vy = 0; this.speed = 0;        // face-widths per second (vy > 0 means rising)
    this.hist = [];
    this.cool = {};
    this.dancing = false;
    this.danceSince = 0;
    this.lastRhythm = 0;
    this.lastActive = 0;
  }

  _cool(name, t, ms) {
    if (t - (this.cool[name] || -1e9) < ms) return false;
    this.cool[name] = t;
    return true;
  }

  /** lm: 478 face landmarks or null. Returns an array of event names that fired. */
  update(t, lm, { aspect = 4 / 3, phoneMoving = false } = {}) {
    const events = [];
    if (!lm) {
      // the landmarker drops the face during fast moves, so keep the last sample around (gap-tolerant)
      this.present = false;
      if (!this.away && this.lastSeen && t - this.lastSeen > 2600) { this.away = true; events.push('away'); }
      return events;
    }
    const size = dist(lm[EYE_R], lm[EYE_L], aspect);
    const cx = (lm[EYE_R].x + lm[EYE_L].x + lm[NOSE].x) / 3;
    const cy = (lm[EYE_R].y + lm[EYE_L].y + lm[NOSE].y) / 3;
    this.present = true;
    this.lastSeen = t;
    if (this.away) { this.away = false; events.push('back'); this.fast = this.slow = size; this.prev = null; this.hist.length = 0; }
    else if (!this.everSeen) { this.everSeen = true; events.push('found'); }

    if (!this.fast) this.fast = this.slow = size;
    const dtRaw = this.prev ? (t - this.prev.t) / 1000 : 0;
    const continuous = this.prev && dtRaw < 0.7;
    const dt = Math.max(dtRaw, 0.016);
    this.fast += (size - this.fast) * (1 - Math.exp(-Math.min(dt, 0.4) / 0.35));
    this.slow += (size - this.slow) * (1 - Math.exp(-Math.min(dt, 0.4) / 4));

    if (continuous) {
      const fw = Math.max(this.size, 1e-3);
      const vy = -(cy - this.prev.cy) / dt / fw;                                           // up is positive
      const vx = ((cx - this.prev.cx) * aspect) / dt / fw;
      this.vy += (vy - this.vy) * 0.6;
      this.speed += (Math.hypot(vx, vy) - this.speed) * 0.6;
    }
    this.hist.push({ t, cx, cy });
    while (this.hist.length && t - this.hist[0].t > 3200) this.hist.shift();
    this.size = size;
    this.cx = cx;
    this.cy = cy;
    this.prev = { t, cx, cy };
    if (this.speed > 0.35) this.lastActive = t;

    if (phoneMoving) return events;

    const ratio = this.fast / (this.slow || size);
    if (ratio > 1.26 && this._cool('approach', t, 6000)) events.push('approach');
    else if (ratio < 0.76 && this._cool('retreat', t, 6000)) events.push('retreat');

    // jump: the face is more than ~a face-width higher than it was half a second ago (works even if the
    // landmarker lost the face for a few frames in between)
    let rise = 0;
    for (const h of this.hist) if (t - h.t < 520 && t - h.t > 30) rise = Math.max(rise, (h.cy - cy) / Math.max(size, 1e-3));
    if (rise > 1.1 && this._cool('jump', t, 2500)) events.push('jump');

    // sudden movement: a spike of speed in any direction
    if (this.speed > 9 && this._cool('sudden', t, 9000)) events.push('sudden');

    // dancing: rhythmic bobbing or swaying for a couple of seconds
    const rhythm = this._rhythm(t, size, aspect);
    if (rhythm) this.lastRhythm = t;
    if (rhythm && !this.dancing) {
      if (!this.danceSince) this.danceSince = t;
      if (t - this.danceSince > 1800) { this.dancing = true; events.push('dance-start'); }
    } else if (!rhythm && t - this.lastRhythm > 2200) {
      this.danceSince = 0;
      if (this.dancing) { this.dancing = false; events.push('dance-end'); }
    }
    return events;
  }

  _rhythm(t, size, aspect) {
    const h = this.hist.filter((p) => t - p.t < 3000);
    if (h.length < 12) return false;
    const reversals = (key, minAmp) => {
      let dir = 0, ext = h[0][key], n = 0;
      for (const p of h) {
        const d = p[key] - ext;
        if (Math.abs(d) * (key === 'cx' ? aspect : 1) / Math.max(size, 1e-3) >= minAmp) {
          const nd = d > 0 ? 1 : -1;
          if (dir !== 0 && nd !== dir) n++;
          dir = nd;
          ext = p[key];
        }
      }
      return n;
    };
    return reversals('cy', 0.22) >= 4 || reversals('cx', 0.45) >= 4;
  }
}

// pose landmark indices
const P = { nose: 0, lShoulder: 11, rShoulder: 12, lWrist: 15, rWrist: 16, lHip: 23, rHip: 24 };

/** How energetically the whole body is moving, in torso-lengths per second. */
export class BodyEnergy {
  constructor() { this.prev = null; this.energy = 0; }
  update(t, lm, aspect = 4 / 3) {
    if (!lm) { this.prev = null; this.energy = 0; return 0; }
    const pts = [P.nose, P.lShoulder, P.rShoulder, P.lWrist, P.rWrist, P.lHip, P.rHip].map((i) => lm[i]);
    const sh = { x: (lm[P.lShoulder].x + lm[P.rShoulder].x) / 2, y: (lm[P.lShoulder].y + lm[P.rShoulder].y) / 2 };
    const hip = { x: (lm[P.lHip].x + lm[P.rHip].x) / 2, y: (lm[P.lHip].y + lm[P.rHip].y) / 2 };
    const torso = Math.max(dist(sh, hip, aspect), 0.08);
    if (this.prev) {
      const dt = Math.max((t - this.prev.t) / 1000, 0.05);
      let sum = 0;
      pts.forEach((p, i) => { sum += dist(p, this.prev.pts[i], aspect); });
      const e = sum / pts.length / torso / dt;
      this.energy += (e - this.energy) * 0.5;
    }
    this.prev = { t, pts: pts.map((p) => ({ x: p.x, y: p.y })) };
    return this.energy;
  }
}
