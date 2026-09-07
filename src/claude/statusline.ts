import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { contextDirFor } from '../context/node-file.js';
import type { GraphV1 } from '../graph/types.js';
import { readWorkspace } from '../graph/workspace.js';
import { renderStatusline, renderSubagent } from './format.js';
import { readStats, readSession, emptyStats, resolveContextDir, type Stats } from './state.js';
import { readWiring, readWiringAt, computeStats } from './stats.js';

/**
 * The statusline's fast path is the hook-maintained cache (graft/.cache/stats.json).
 * When it's absent — a fresh checkout, or a plain `graft build` that doesn't write the
 * cache — fall back to reading the graph itself (wiring.json) so the bar reflects reality
 * immediately instead of showing "not built". An empty wiring.json is still a graph
 * (docs-only repos legitimately have 0 nodes); "not built" is only when the artifact
 * is missing. The graph carries no drift signal, so it reads as synced until the next
 * edit repopulates the cache. Still a pure read (no subprocess): the cache is preferred
 * because it carries live dirty/stale state.
 *
 * A workspace parent (graft/workspace.json) has no wiring.json of its own — its nodes
 * live in the children — so its stats are the sum of every built child's graph. A
 * parent whose children are all unbuilt is still "not built": `graft build` there
 * builds the children, so the hint is right.
 */
export function resolveStats(dir: string): Stats | null {
  const cached = readStats(dir);
  if (cached && cached.nodeCount > 0) return cached;
  const wiring = readWiring(dir);
  if (wiring) return { ...emptyStats(), ...computeStats(wiring) };
  const ws = readWorkspace(dir, resolveContextDir(dir));
  if (ws) return workspaceStats(dir, ws.children);
  return null;
}

/** Sum of the built children's graphs, or null when none is built. Children read their
 * default `graft/` (as `loadWorkspaceGraphs` does), never the parent's GRAFT_DIR override. */
function workspaceStats(root: string, children: readonly string[]): Stats | null {
  const graphs = children.map((c) => readWiringAt(contextDirFor(join(root, c)))).filter((g): g is GraphV1 => g !== null);
  if (graphs.length === 0) return null;
  const sum = emptyStats();
  const languages = new Set<string>();
  for (const g of graphs) {
    const s = computeStats(g);
    sum.nodeCount += s.nodeCount;
    sum.edgeCount += s.edgeCount;
    sum.totalCount += s.totalCount;
    sum.readyCount += s.readyCount;
    for (const l of s.languages) languages.add(l);
  }
  sum.languages = [...languages].sort();
  return sum;
}

export function main(): void {
  let input: any = {};
  try { input = JSON.parse(readFileSync(0, 'utf8')); } catch { /* no/invalid stdin */ }
  const dir = process.env.CLAUDE_PROJECT_DIR || input.cwd || process.cwd();
  const session = readSession(dir, input.session_id || 'default');
  const agent = input?.agent?.name;
  if (agent) { process.stdout.write(renderSubagent(agent, session)); return; }
  const stats = resolveStats(dir);
  const raw = input?.context_window?.used_percentage;
  const ctxPct = typeof raw === 'number' ? Math.round(raw) : null;
  const str = (v: unknown): string | undefined => (typeof v === 'string' && v ? v : undefined);
  process.stdout.write(renderStatusline(stats, session, {
    ctxPct,
    modelId: str(input?.model?.id),
    modelName: str(input?.model?.display_name),
    effort: str(input?.effort?.level),
    cwd: str(input?.cwd) ?? str(input?.workspace?.current_dir) ?? process.cwd(),
  }).join('\n'));
}
