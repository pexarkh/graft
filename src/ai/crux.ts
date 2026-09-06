/**
 * Tier-2 "meaning" call for the code graph — batched one request per file.
 *
 * Given a source file (with 1-based line numbers) and the list of definitions in
 * it, one call returns, for each definition:
 *   1. `summary` — one plain-English sentence: what the symbol is *for*, at the
 *      business-logic level, not a restatement of its signature.
 *   2. `crux_start`/`crux_end` — the smallest contiguous range of FILE line
 *      numbers (inside that symbol's own span) that a reviewer must read to see
 *      the decision or rule the code encodes. `0/0` means there is no single
 *      crux (a trivial getter, a plain data holder).
 *
 * Batching per file means N definitions cost one request, not N — and the model
 * sees each symbol's neighbours, which sharpens the summaries. Line numbers are
 * consumed once, at write time, to slice the crux text verbatim from source.
 */
import { envPositiveInt, type ChatModel, type ChatResponse } from "./llm/types.js";
import { recoverToolArgsFromContent, warnToolChoiceIgnored } from "./llm/recover-tool.js";
import type { Kind } from "../graph/types.js";

/** One definition we want described, located by its line span within the file. */
export interface NodeRef {
  id: string;
  kind: Kind;
  signature: string | null;
  startLine: number; // 1-based file line where the definition starts
  endLine: number;
}

export interface FileCruxInput {
  path: string;
  source: string;
  nodes: NodeRef[];
}

export interface NodeCrux {
  id: string;
  summary: string;
  crux_start: number; // file line, within the symbol's span; 0 = no distinct crux
  crux_end: number;
}

export interface CruxSummarizer {
  describeFile(input: FileCruxInput): Promise<NodeCrux[]>;
  /** Set by {@link ChatCruxSummarizer} after each call; optional on fakes. */
  lastMiss?: CruxMiss | null;
}

/** Why a crux call produced no usable summaries (#235). */
export type CruxMissKind = "empty-toolCalls" | "unparseable" | "truncated" | "empty-parsed";

export interface CruxMiss {
  kind: CruxMissKind;
  finishReason: string | null;
}

function isTruncatedStop(reason: string | null): boolean {
  if (!reason) return false;
  const r = reason.toLowerCase();
  return r === "length" || r === "max_tokens";
}

/** Classify an empty/unusable crux reply. `null` means at least one usable summary. */
export function classifyCruxMiss(res: ChatResponse, parsed: NodeCrux[]): CruxMiss | null {
  const finishReason = res.stopReason;
  if (parsed.some((p) => p.summary.trim())) return null;
  if (isTruncatedStop(finishReason)) return { kind: "truncated", finishReason };
  if (parsed.length > 0) return { kind: "empty-parsed", finishReason };
  const emptyTools = res.toolCalls.length === 0;
  const emptyText = !res.text?.trim();
  if (emptyTools && emptyText) return { kind: "empty-toolCalls", finishReason };
  return { kind: "unparseable", finishReason };
}

/** Per-file error text: miss class + the provider's finish_reason (#235). */
export function formatCruxMiss(kind: CruxMissKind, finishReason: string | null): string {
  const fr = finishReason == null || finishReason === "" ? "null" : finishReason;
  return `model returned no usable symbol summaries [${kind}, finish_reason=${fr}]`;
}

const SYSTEM_PROMPT = `You explain code definitions for a code graph that helps engineers navigate a codebase.

You are given ONE source file with 1-based line numbers, and a list of TARGET definitions in it. Describe EVERY target via the record_symbols tool.

Rules:
- Return EXACTLY ONE entry for EVERY target id, using that id verbatim. The number of entries you return MUST equal the number of targets. Never omit a target: a reply missing any id is invalid and will be re-requested.
- A trivial symbol is NOT an exception. You still return it — with a one-sentence summary and crux 0/0 (see below). "Skip" means "give it no crux span", NEVER "leave it out".
- summary: ONE sentence — what the symbol is FOR at the business-logic level (the problem it solves or the rule it enforces), not a restatement of its signature.
- crux_start / crux_end: FILE line numbers (as shown), inside that symbol's own line range. Pick the SINGLE most important contiguous span — the core branch, formula, guard, or state change — at most ~8 lines, and NEVER the whole function. When there is no single focal span (a trivial getter, a plain data holder, a one-line delegation, or logic spread evenly), use crux_start: 0 and crux_end: 0. That 0/0 IS the answer — do not drop the entry.`;

const RECORD_TOOL = "record_symbols";

const SYMBOLS_SCHEMA = {
  type: "object",
  properties: {
    symbols: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" },
          summary: { type: "string" },
          crux_start: { type: "number" },
          crux_end: { type: "number" },
        },
        required: ["id", "summary", "crux_start", "crux_end"],
      },
    },
  },
  required: ["symbols"],
} as const;

/** Cap on the code sent per request. With windowing (below) it only bites when
 * one window — a single giant symbol — exceeds it. Env: GRAFT_CRUX_MAX_CHARS. */
export const DEFAULT_CRUX_MAX_CHARS = 18_000;
export function cruxMaxChars(): number {
  return envPositiveInt("GRAFT_CRUX_MAX_CHARS", DEFAULT_CRUX_MAX_CHARS);
}

/** Targets described per request. Bounds the reply so a 120-symbol file cannot
 * overrun the output budget, and bounds the input to the lines those targets
 * cover. Env: GRAFT_CRUX_TARGETS_PER_CALL. */
export const DEFAULT_TARGETS_PER_CALL = 25;
export function targetsPerCall(): number {
  return envPositiveInt("GRAFT_CRUX_TARGETS_PER_CALL", DEFAULT_TARGETS_PER_CALL);
}

/** One request's worth of a file: which targets, and the file-absolute line range shown. */
export interface CruxWindow {
  nodes: NodeRef[];
  startLine: number;
  endLine: number;
}

/**
 * Split a file's targets into windows of ≤ `perCall`, ordered by start line, each
 * carrying only the lines that cover its symbols. Before this, one request sent the
 * first 18K chars and asked about EVERY symbol — on a 2,600-line file ~100 targets
 * pointed at code the model could not see. Line numbers stay file-absolute, so the
 * prompt's rules and the returned crux spans need no translation. A file-level node
 * spans the whole file; it rides in the first window, which is extended to start at
 * line 1 so the module head (imports, constants) is in view for it.
 */
export function cruxWindows(input: FileCruxInput, perCall = targetsPerCall()): CruxWindow[] {
  const lastLine = input.source.split("\n").length;
  const files = input.nodes.filter((n) => n.kind === "file");
  const syms = input.nodes.filter((n) => n.kind !== "file").sort((a, b) => a.startLine - b.startLine || a.endLine - b.endLine);
  const windows: CruxWindow[] = [];
  const groups = Math.max(1, Math.ceil(syms.length / perCall));
  for (let i = 0; i < groups; i++) {
    const group = syms.slice(i * perCall, (i + 1) * perCall);
    const nodes = i === 0 ? [...files, ...group] : group;
    if (nodes.length === 0) continue;
    let startLine = group.length ? Math.max(1, Math.min(...group.map((n) => n.startLine))) : 1;
    const endLine = group.length ? Math.min(lastLine, Math.max(...group.map((n) => n.endLine))) : lastLine;
    if (i === 0 && files.length) startLine = 1;
    windows.push({ nodes, startLine, endLine: Math.max(startLine, endLine) });
  }
  return windows;
}

function numberLines(lines: string[], fromLine: number): string {
  const max = cruxMaxChars();
  let text = lines.map((line, i) => `${fromLine + i}\t${line}`).join("\n");
  if (text.length > max) text = `${text.slice(0, max)}\n… (truncated)`;
  return text;
}

function userContent(path: string, lines: string[], w: CruxWindow): string {
  const targets = w.nodes
    .map(
      (n) =>
        `- id=${n.id} | ${n.kind} | lines L${n.startLine}-L${n.endLine}` +
        (n.signature ? ` | ${n.signature}` : ""),
    )
    .join("\n");
  const n = w.nodes.length;
  const partial = w.startLine > 1 || w.endLine < lines.length;
  const shown = partial ? ` (showing lines L${w.startLine}-L${w.endLine} of ${lines.length})` : "";
  const body = numberLines(lines.slice(w.startLine - 1, w.endLine), w.startLine);
  return `FILE: ${path}${shown}\n\n${body}\n\nTARGETS (${n} — return all ${n}, one entry per id):\n${targets}`;
}

/**
 * Some models (Qwen3-Coder via Bedrock, for one) echo the WHOLE target line as the
 * id — `src/x.ts#f | function | lines L5-L10 | function f()` — instead of the bare
 * id. Every entry then matches no target, and the file is reported as a miss even
 * though the summaries were fine. The ` | ` separator is ours (see `userContent`)
 * and never occurs inside an id, so keep only what precedes it; also drop a copied
 * `id=` prefix.
 */
export function normalizeSymbolId(raw: string): string {
  return raw.split(" | ")[0].trim().replace(/^id=/, "");
}

/** Normalize the tool's parsed argument object into a {@link NodeCrux} list. */
function parseResults(obj: { symbols?: unknown } | undefined): NodeCrux[] {
  if (!obj || !Array.isArray(obj.symbols)) return [];
  const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? Math.trunc(v) : 0);
  return obj.symbols
    .map((s) => s as Record<string, unknown>)
    .filter((s) => typeof s.id === "string")
    .map((s) => ({
      id: normalizeSymbolId(s.id as string),
      summary: typeof s.summary === "string" ? s.summary.trim() : "",
      crux_start: num(s.crux_start),
      crux_end: num(s.crux_end),
    }));
}

/**
 * Some OpenAI-compatible gateways ignore forced `tool_choice` and put the tool
 * payload in `content` instead (plain `{symbols:…}`, fenced JSON, or an emulated
 * `[{name, parameters}]` array). Without this recovery the meaning pass sees an
 * empty `toolCalls` list, leaves every node `pending`, and `graft check` loops
 * on "run --deep" forever (#172; same trigger as #129 for the crux path).
 */
function argsFromResponse(res: { text: string; toolCalls: { name: string; args: unknown }[] }): {
  symbols?: unknown;
} | undefined {
  const call = res.toolCalls.find((c) => c.name === RECORD_TOOL) ?? res.toolCalls[0];
  if (call?.args && typeof call.args === "object" && !Array.isArray(call.args)) {
    return call.args as { symbols?: unknown };
  }
  const recovered = recoverToolArgsFromContent(res.text, {
    toolNames: [RECORD_TOOL, "emit_json"],
    payloadKey: "symbols",
  });
  if (!recovered) warnToolChoiceIgnored("crux", res.text?.trim() ? "unparsed" : "empty");
  return recovered as { symbols?: unknown } | undefined;
}

/** Crux summarizer backed by any {@link ChatModel} via forced tool calling. */
export class ChatCruxSummarizer implements CruxSummarizer {
  lastMiss: CruxMiss | null = null;

  constructor(private model: ChatModel) {}

  /** One request per window (see {@link cruxWindows}); results are concatenated and
   * `lastMiss` records the first window that produced nothing usable. */
  async describeFile(input: FileCruxInput): Promise<NodeCrux[]> {
    this.lastMiss = null;
    if (input.nodes.length === 0) return [];
    const lines = input.source.split("\n");
    const out: NodeCrux[] = [];
    for (const w of cruxWindows(input)) {
      const res = await this.model.create({
        temperature: 0,
        maxTokens: 8192,
        tools: [
          {
            name: RECORD_TOOL,
            description: "Record each target definition's purpose and crux line range.",
            parameters: SYMBOLS_SCHEMA as unknown as Record<string, unknown>,
          },
        ],
        responseFormat: { kind: "tool", name: RECORD_TOOL },
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: userContent(input.path, lines, w) },
        ],
      });
      const parsed = parseResults(argsFromResponse(res));
      this.lastMiss ??= classifyCruxMiss(res, parsed);
      out.push(...parsed);
    }
    return out;
  }
}
