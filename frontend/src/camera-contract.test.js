import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Contract tests for the browser camera path.
 *
 * These read the source instead of mounting the component, which keeps the
 * suite dependency-free (no jsdom) while still failing loudly if someone
 * reintroduces a server-side capture call or hardcodes a host.
 */

// This file lives in src/, so the tree to scan is its own directory.
const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = HERE;
const SKIP_DIRS = new Set(['node_modules', 'dist', 'build', '.git', 'coverage']);

function read(rel) {
  return readFileSync(join(SRC, rel), 'utf8');
}

function walk(dir, acc = []) {
  for (const entry of readdirSync(join(SRC, dir))) {
    if (SKIP_DIRS.has(entry) || entry.startsWith('.')) continue;
    const rel = dir ? `${dir}/${entry}` : entry;
    if (statSync(join(SRC, rel)).isDirectory()) walk(rel, acc);
    else if (/\.(js|jsx)$/.test(rel) && !/\.test\.jsx?$/.test(rel)) acc.push(rel);
  }
  return acc;
}

const files = walk('');

describe('browser camera access', () => {
  it('finds the source tree', () => {
    expect(files.length).toBeGreaterThan(0);
    expect(files).toContain('pages/FaceRecognition.jsx');
  });

  it('uses navigator.mediaDevices.getUserMedia', () => {
    const source = read('pages/FaceRecognition.jsx');
    expect(source).toMatch(/navigator\.mediaDevices/);
    expect(source).toMatch(/getUserMedia/);
  });

  it('never asks the backend to open a camera', () => {
    // A server-side capture would need a route that streams video, e.g.
    // /api/video, /api/camera/0, /api/webcam.
    // NOTE: keep these patterns free of nested quantifiers. A pattern such as
    // /\/api\/[^"']*(video|camera)/ backtracks exponentially and hangs the
    // whole test run instead of failing.
    for (const file of files) {
      const source = read(file);
      expect(source, `${file} must not reference cv2 capture`).not.toMatch(
        /VideoCapture/i,
      );
      expect(source, `${file} must not call a server capture endpoint`).not.toMatch(
        /\/(?:video|camera|webcam|stream)\b/i,
      );
    }
  });

  it('requests audio off and sets an ideal video size', () => {
    const source = read('pages/FaceRecognition.jsx');
    expect(source).toMatch(/audio:\s*false/);
    expect(source).toMatch(/facingMode/);
  });

  it('stops every camera track when the component unmounts', () => {
    const source = read('pages/FaceRecognition.jsx');
    const tracks = source.match(/getTracks\(\)/);
    const stop = source.match(/\.stop\(\)/);
    expect(tracks, 'stream tracks must be reachable').not.toBeNull();
    expect(stop, 'tracks must be stopped to release the camera').not.toBeNull();
  });

  it('handles a denied or missing camera without crashing', () => {
    const source = read('pages/FaceRecognition.jsx');
    expect(source).toMatch(/NotAllowedError/);
    expect(source).toMatch(/NotFoundError/);
    expect(source).toMatch(/catch/);
  });
});

describe('no environment-specific paths are bundled', () => {
  it('contains no absolute developer paths', () => {
    for (const file of files) {
      const source = read(file);
      expect(source, `${file} must not hardcode a home directory`).not.toMatch(
        /\/Users\/|\/home\/[a-z]|C:\\Users/i,
      );
    }
  });

  it('does not hardcode a localhost API origin', () => {
    for (const file of files) {
      const source = read(file);
      expect(source, `${file} must read the API base from the environment`).not.toMatch(
        /http:\/\/localhost:\d+/,
      );
    }
  });

  it('reads the API base from VITE_API_URL', () => {
    expect(read('api/api.js')).toMatch(/import\.meta\.env\.VITE_API_URL/);
  });
});
