/**
 * The shim's resolution behaviour, exercised by actually running it.
 *
 * This is the regression test for the "installed graft once, still on the old
 * version" report: the shim used to take the FIRST candidate that existed, and
 * the first candidate is the absolute path baked in at `graft init` time. So
 * `npm i -g @nanonets/graft@latest` upgraded a directory the shim never looked
 * at, and the user's hooks kept loading whatever version wired the repo. The
 * shim now takes the highest-versioned candidate instead.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { hooksShim } from '../src/claude/shim-template.js';
import { tmpRepo } from './helpers.js';

// Fake versions sit at 99.x on purpose: the shim also sees whatever @nanonets/graft is
// REALLY installed on the machine (via execPath/../lib and `npm root -g`), and it takes
// the highest version among every candidate. Realistic fake versions lose to a real
// 0.16.0 install, the real hooks entry runs, no marker is written, and the test fails
// for a reason that has nothing to do with the shim.
const NEWER = '99.1.0';
const OLDER = '99.0.1';

/** A fake installed @nanonets/graft whose hooks entry records that it ran. */
function fakeInstall(root: string, name: string, version: string): string {
  const pkg = join(root, name);
  const distClaude = join(pkg, 'dist', 'claude');
  mkdirSync(distClaude, { recursive: true });
  writeFileSync(join(pkg, 'package.json'), JSON.stringify({ name: '@nanonets/graft', version }));
  // CJS on purpose: no "type" field, so `import()` hands back module.exports and
  // `m.main(...)` resolves — same shape the real dist has for the shim's call.
  writeFileSync(
    join(distClaude, 'hooks.js'),
    `module.exports.main = () => require('node:fs').writeFileSync(process.env.MARKER, ${JSON.stringify(version)});\n`,
  );
  return distClaude;
}

/** Runs the shim with the given baked dir and project dir; returns the version
 * of the install that actually got loaded (or null if none did). */
function runShim(root: string, bakedDir: string, projectDir: string): string | null {
  const shimPath = join(root, 'graft-hooks.cjs');
  const marker = join(root, 'loaded.txt');
  writeFileSync(shimPath, hooksShim(bakedDir));
  const res = spawnSync(process.execPath, [shimPath, 'session-start'], {
    encoding: 'utf8',
    env: { ...process.env, MARKER: marker, CLAUDE_PROJECT_DIR: projectDir },
  });
  assert.equal(res.status, 0, `shim exited ${res.status}: ${res.stderr}`);
  return existsSync(marker) ? readFileSync(marker, 'utf8') : null;
}

test('an upgraded global install wins over the stale baked path', () => {
  const root = tmpRepo('shim-upgrade');
  const stale = fakeInstall(root, 'old-node-install', OLDER);
  fakeInstall(join(root, 'project', 'node_modules', '@nanonets'), 'graft', NEWER);
  // BAKED points at the install that `graft init` ran from — still on disk (an
  // nvm switch leaves it there), still first in the candidate list, now stale.
  assert.equal(runShim(root, stale, join(root, 'project')), NEWER);
});

test('the baked path still wins when it is the newest', () => {
  const root = tmpRepo('shim-baked-newest');
  const baked = fakeInstall(root, 'current', NEWER);
  fakeInstall(join(root, 'project', 'node_modules', '@nanonets'), 'graft', OLDER);
  assert.equal(runShim(root, baked, join(root, 'project')), NEWER);
});

test('a single candidate is used whatever its version', () => {
  const root = tmpRepo('shim-single');
  const only = fakeInstall(root, 'only', OLDER);
  mkdirSync(join(root, 'project'), { recursive: true });
  assert.equal(runShim(root, only, join(root, 'project')), OLDER);
});

test('an install with an unreadable version loses to a known one', () => {
  const root = tmpRepo('shim-noversion');
  const broken = fakeInstall(root, 'broken', '0.0.0');
  writeFileSync(join(root, 'broken', 'package.json'), 'not json');
  fakeInstall(join(root, 'project', 'node_modules', '@nanonets'), 'graft', OLDER);
  assert.equal(runShim(root, broken, join(root, 'project')), OLDER);
});

test('no candidate at all exits quietly — a hook must never fail the session', () => {
  const root = tmpRepo('shim-none');
  mkdirSync(join(root, 'project'), { recursive: true });
  assert.equal(runShim(root, join(root, 'nowhere'), join(root, 'project')), null);
});
