// Replays inputs through the browser playground's JS modules and prints JSON results.
// Used by tests/test_web_parity.py to check web/*.js against the Python implementation.
//
//   node tests/js/replay.mjs tracker <fixture.json> <highThresh> [iouBuffer] [stride]
//   node tests/js/replay.mjs postprocess <case.json>
import fs from 'node:fs';
import { postprocessDetr } from '../../web/detector.js';
import { ByteTracker } from '../../web/tracker.js';

const [mode, path, arg, arg2, arg3] = process.argv.slice(2);
const input = JSON.parse(fs.readFileSync(path, 'utf8'));

if (mode === 'tracker') {
  const thr = Number(arg);
  const out = {};
  for (const [name, frames] of Object.entries(input.sequences)) {
    const tracker = new ByteTracker({
      highThresh: thr,
      lowThresh: Math.max(0.1, thr * 0.35),
      newTrackThresh: Math.min(0.95, thr + 0.05),
      iouBuffer: Number(arg2 ?? 0),
    });
    const stride = Number(arg3 ?? 1);
    out[name] = frames.filter((_, i) => i % stride === 0).map(dets => {
      const tracks = tracker.update(dets.map(([xmin, ymin, xmax, ymax, score]) => ({
        box: { xmin, ymin, xmax, ymax }, score, label: 'person', classId: 0,
      })));
      const fmt = t => [t.id, t.box.xmin, t.box.ymin, t.box.xmax, t.box.ymax];
      return { tracks: tracks.map(fmt), coasting: tracker.coastingTracks(3).map(fmt) };
    });
  }
  process.stdout.write(JSON.stringify(out));
} else if (mode === 'postprocess') {
  const dets = postprocessDetr({
    logits: Float32Array.from(input.logits),
    logitsDims: input.dims,
    boxes: Float32Array.from(input.boxes),
    id2label: input.id2label,
    width: input.width,
    height: input.height,
    threshold: input.threshold,
    allowedLabels: input.allowed ? new Set(input.allowed) : null,
  });
  process.stdout.write(JSON.stringify(dets.map(d => [d.classId, d.score, d.box.xmin, d.box.ymin, d.box.xmax, d.box.ymax])));
} else {
  throw new Error(`unknown mode ${mode}`);
}
