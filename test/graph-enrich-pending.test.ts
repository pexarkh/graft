/**
 * #172: `graft build --deep` can finish with every node still `pending` when the
 * meaning pass gets an empty reply (no tool call / no symbols) — and re-running
 * does not clear it, while `graft check` only says "run --deep".
 *
 * The #127 gate already catches thrown provider errors; this covers the silent
 * empty-success path that left CI stuck on an unresolvable check note.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { enrichGraph } from "../src/graph/enrich.js";
import { ChatCruxSummarizer, cruxWindows } from "../src/ai/crux.js";
import { ChatSummarizer } from "../src/ai/summarize.js";
import { formatGraphCheckReport, type GraphCheckResult } from "../src/graph/check.js";
import type { CruxSummarizer, FileCruxInput, NodeCrux } from "../src/ai/crux.js";
import type { ChatModel, ChatRequest, ChatResponse } from "../src/ai/llm/types.js";
import type { NodeV1 } from "../src/graph/types.js";

/** Returns no symbols — the silent failure mode behind #172. */
class EmptyCrux implements CruxSummarizer {
  calls = 0;
  async describeFile(_input: FileCruxInput): Promise<NodeCrux[]> {
    this.calls++;
    return [];
  }
}

/** Returns entries whose summaries are empty strings. */
class BlankSummaryCrux implements CruxSummarizer {
  async describeFile(input: FileCruxInput): Promise<NodeCrux[]> {
    return input.nodes.map((n) => ({ id: n.id, summary: "  ", crux_start: 0, crux_end: 0 }));
  }
}

function node(path: string, name: string, kind: NodeV1["kind"] = "function"): NodeV1 {
  return {
    id: kind === "file" ? path : `${path}#${name}`,
    name,
    kind,
    path,
    span: "L1-L3",
    signature: kind === "file" ? null : `${name}()`,
    exported: true,
    origin: "ast",
    body_hash: `h-${name}`,
    summary_state: "pending",
    summary: null,
    crux: null,
  };
}

function calcFixture(): { nodes: NodeV1[]; sources: Map<string, string> } {
  const path = "src/calc.py";
  return {
    nodes: [
      node(path, "calc.py", "file"),
      node(path, "add"),
      node(path, "mul"),
      node(path, "total"),
    ],
    sources: new Map([[path, "def add(a, b):\n  return a + b\n"]]),
  };
}

test("#172: an empty meaning reply is a failed file, not a silent all-pending success", async () => {
  const { nodes, sources } = calcFixture();
  const summarizer = new EmptyCrux();

  const stats = await enrichGraph(nodes, new Map(), sources, { summarizer, concurrency: 1 });

  assert.equal(stats.computed, 0);
  assert.equal(stats.pending, 4);
  assert.equal(stats.failedFiles, 1, "empty reply must count as a failed file so the CLI can exit non-zero");
  assert.ok(stats.errors.length > 0, "empty reply must leave a readable error");
  assert.match(stats.errors[0] ?? "", /no symbol|empty/i);
  // collectFileCrux retries once when the model omits every id
  assert.equal(summarizer.calls, 2);
  for (const n of nodes) assert.equal(n.summary_state, "pending");
});

test("#172: empty summaries must not be cached as ready — a re-run still retries them", async () => {
  const { nodes, sources } = calcFixture();

  const first = await enrichGraph(nodes, new Map(), sources, {
    summarizer: new BlankSummaryCrux(),
    concurrency: 1,
  });
  assert.equal(first.computed, 0, "blank summaries are not a successful compute");
  assert.equal(first.pending, 4);
  for (const n of nodes) {
    assert.equal(n.summary_state, "pending", "blank summary must not flip the node to ready");
    assert.equal(n.summary, null);
  }

  // Prior graph still has pending+null; a working summarizer on the next --deep must retry.
  const prior = new Map(nodes.map((n) => [n.id, structuredClone(n)]));
  const { nodes: again } = calcFixture();
  let calls = 0;
  const working: CruxSummarizer = {
    async describeFile(input) {
      calls++;
      return input.nodes.map((n) => ({
        id: n.id,
        summary: `purpose of ${n.id}`,
        crux_start: 0,
        crux_end: 0,
      }));
    },
  };
  const second = await enrichGraph(again, prior, sources, { summarizer: working, concurrency: 1 });
  assert.equal(calls, 1, "pending nodes from a blank pass must be retried");
  assert.equal(second.computed, 4);
  assert.equal(second.pending, 0);
  assert.equal(second.cached, 0);
});

test("#172: check names pending nodes and does not pretend re-running --deep always clears them", () => {
  const r: GraphCheckResult = {
    ok: true,
    missing: false,
    added: [],
    removed: [],
    changed: [],
    stale: [],
    pending: 4,
    pendingIds: ["src/calc.py", "src/calc.py#add", "src/calc.py#mul", "src/calc.py#total"],
    nodes: 4,
  };
  const report = formatGraphCheckReport(r);
  assert.match(report, /src\/calc\.py#add/);
  assert.match(report, /pending/);
  // Must not be the old dead-end that only says "run graft build --deep".
  assert.match(report, /already|failed|errors|meaning pass/i);
});

test("#172: ChatCruxSummarizer recovers symbols when the gateway puts the tool payload in content", async () => {
  class ContentOnlyModel implements ChatModel {
    readonly label = "fake:content-tool";
    async create(_req: ChatRequest): Promise<ChatResponse> {
      const payload = {
        symbols: [{ id: "src/calc.py#add", summary: "adds two numbers", crux_start: 0, crux_end: 0 }],
      };
      // DeepSeek-style: ignore tool_choice, emulate the call in content (#129 trigger).
      return {
        text: JSON.stringify([{ name: "record_symbols", parameters: payload }]),
        toolCalls: [],
        usage: { input: 0, output: 0, cacheRead: 0, cacheCreate: 0 },
        stopReason: "stop",
        assistant: { role: "assistant", content: "" },
      };
    }
  }

  const out = await new ChatCruxSummarizer(new ContentOnlyModel()).describeFile({
    path: "src/calc.py",
    source: "def add(a, b):\n  return a + b\n",
    nodes: [{ id: "src/calc.py#add", kind: "function", signature: null, startLine: 1, endLine: 2 }],
  });
  assert.equal(out.length, 1);
  assert.equal(out[0].id, "src/calc.py#add");
  assert.equal(out[0].summary, "adds two numbers");
});

test("ChatCruxSummarizer maps an id echoed as the whole target line back to the bare id", async () => {
  class EchoingIdModel implements ChatModel {
    readonly label = "fake:echo-id";
    async create(_req: ChatRequest): Promise<ChatResponse> {
      // Qwen3-Coder on Bedrock copied everything after `id=` from the target line.
      const args = {
        symbols: [
          { id: "src/calc.py | file | lines L1-L2", summary: "arithmetic helpers", crux_start: 0, crux_end: 0 },
          { id: "id=src/calc.py#add | function | lines L1-L2 | def add(a, b)", summary: "adds two numbers", crux_start: 2, crux_end: 2 },
        ],
      };
      return {
        text: "",
        toolCalls: [{ id: "c1", name: "record_symbols", args }],
        usage: { input: 0, output: 0, cacheRead: 0, cacheCreate: 0 },
        stopReason: "tool_calls",
        assistant: { role: "assistant", content: "" },
      };
    }
  }
  const crux = new ChatCruxSummarizer(new EchoingIdModel());
  const out = await crux.describeFile({
    path: "src/calc.py",
    source: "def add(a, b):\n  return a + b\n",
    nodes: [
      { id: "src/calc.py", kind: "file", signature: null, startLine: 1, endLine: 2 },
      { id: "src/calc.py#add", kind: "function", signature: "def add(a, b)", startLine: 1, endLine: 2 },
    ],
  });
  assert.deepEqual(out.map((o) => o.id), ["src/calc.py", "src/calc.py#add"]);
  assert.equal(out[1].summary, "adds two numbers");
  assert.equal(crux.lastMiss, null);
});

/** 200 one-line functions plus the file node — the shape of a big router module. */
function bigFile(): FileCruxInput {
  const n = 200;
  const source = Array.from({ length: n }, (_, i) => `def f${i}(): return ${i}`).join("\n");
  const nodes: FileCruxInput["nodes"] = [{ id: "big.py", kind: "file", signature: null, startLine: 1, endLine: n }];
  for (let i = 0; i < n; i++) nodes.push({ id: `big.py#f${i}`, kind: "function", signature: `def f${i}()`, startLine: i + 1, endLine: i + 1 });
  return { path: "big.py", source, nodes };
}

/** Answers exactly the targets listed in the prompt it was shown, and keeps every prompt. */
class WindowEchoModel implements ChatModel {
  readonly label = "fake:window-echo";
  prompts: string[] = [];
  async create(req: ChatRequest): Promise<ChatResponse> {
    const user = req.messages.find((m) => m.role === "user")!.content;
    this.prompts.push(user);
    const ids = [...user.matchAll(/^- id=(\S+) \|/gm)].map((m) => m[1]);
    const args = { symbols: ids.map((id) => ({ id, summary: `about ${id}`, crux_start: 0, crux_end: 0 })) };
    return {
      text: "",
      toolCalls: [{ id: "c", name: "record_symbols", args }],
      usage: { input: 0, output: 0, cacheRead: 0, cacheCreate: 0 },
      stopReason: "tool_calls",
      assistant: { role: "assistant", content: "" },
    };
  }
}

test("crux windows: ≤25 targets per call, each shown only its own lines, file node rides in the first window", async () => {
  const input = bigFile();
  const windows = cruxWindows(input, 25);
  assert.equal(windows.length, 8);
  assert.deepEqual(windows.map((w) => [w.startLine, w.endLine]).slice(0, 3), [[1, 25], [26, 50], [51, 75]]);
  assert.equal(windows[0].nodes.length, 26, "file node + 25 symbols");
  assert.equal(windows[0].nodes[0].id, "big.py");
  const model = new WindowEchoModel();
  const crux = new ChatCruxSummarizer(model);
  const out = await crux.describeFile(input);
  assert.equal(model.prompts.length, 8);
  assert.equal(out.length, 201);
  assert.deepEqual(new Set(out.map((o) => o.id)), new Set(input.nodes.map((n) => n.id)));
  assert.equal(crux.lastMiss, null);
  assert.match(model.prompts[1], /\(showing lines L26-L50 of 200\)/);
  assert.match(model.prompts[1], /\n26\tdef f25\(\)/, "numbering stays file-absolute");
  assert.doesNotMatch(model.prompts[1], /def f0\(\)/, "a window carries only its own lines");
  assert.match(model.prompts[1], /TARGETS \(25 — return all 25/);
});

test("crux windows: GRAFT_CRUX_TARGETS_PER_CALL overrides the window size", () => {
  const prev = process.env.GRAFT_CRUX_TARGETS_PER_CALL;
  process.env.GRAFT_CRUX_TARGETS_PER_CALL = "100";
  try {
    assert.equal(cruxWindows(bigFile()).length, 2);
  } finally {
    if (prev === undefined) delete process.env.GRAFT_CRUX_TARGETS_PER_CALL;
    else process.env.GRAFT_CRUX_TARGETS_PER_CALL = prev;
  }
  assert.equal(cruxWindows(bigFile()).length, 8);
});

test("summary cap: GRAFT_SUMMARY_MAX_CHARS controls where the file is cut", async () => {
  let seen = "";
  const model: ChatModel = {
    label: "fake:capture",
    async create(req) {
      seen = req.messages.find((m) => m.role === "user")!.content;
      return { text: "ok", toolCalls: [], usage: { input: 0, output: 0, cacheRead: 0, cacheCreate: 0 }, stopReason: "stop", assistant: { role: "assistant", content: "ok" } };
    },
  };
  const code = "x".repeat(30_000);
  const prev = process.env.GRAFT_SUMMARY_MAX_CHARS;
  try {
    delete process.env.GRAFT_SUMMARY_MAX_CHARS;
    await new ChatSummarizer(model).summarize(code, { path: "big.py" });
    assert.match(seen, /truncated at 24000 characters/);
    process.env.GRAFT_SUMMARY_MAX_CHARS = "100000";
    await new ChatSummarizer(model).summarize(code, { path: "big.py" });
    assert.doesNotMatch(seen, /truncated/);
    process.env.GRAFT_SUMMARY_MAX_CHARS = "100";
    await new ChatSummarizer(model).summarize(code, { path: "big.py" });
    assert.match(seen, /truncated at 100 characters/);
  } finally {
    if (prev === undefined) delete process.env.GRAFT_SUMMARY_MAX_CHARS;
    else process.env.GRAFT_SUMMARY_MAX_CHARS = prev;
  }
});
