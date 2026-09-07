/**
 * The Context Graph Engine.
 *
 * Two operations, no database:
 *   - {@link Graft.init}  build `.context/` from a code repo.
 *   - {@link Graft.check} report whether `.context/` is still in
 *     sync with the code (for CI).
 *
 * The graph is a folder of linked markdown files committed to the repo; git is
 * the sync. This class wires the configured LLM provider into the build/check
 * pipelines; an API key is required for any LLM-backed operation.
 */
import { hasPassOverrides, LLM_PASSES, resolveConfig, type EngineConfig, type LlmPass, type ResolvedConfig } from "./ai/providers.js";
import { ChatSynthesizer, type Synthesizer } from "./ai/synthesize.js";
import { ChatSummarizer, type Summarizer } from "./ai/summarize.js";
import { ChatCruxSummarizer, type CruxSummarizer } from "./ai/crux.js";
import { createChatModel } from "./ai/llm/factory.js";
import { emptyUsage, meter, type ChatModel, type UsageTotals } from "./ai/llm/types.js";
import { buildContext, CODE_EXTENSIONS, type BuildProgress, type BuildResult } from "./context/build.js";
import { checkContext, type CheckResult } from "./context/check.js";
import { buildGraph, type GraphBuildOptions, type GraphBuildResult } from "./graph/build.js";
import { checkGraph, type GraphCheckResult } from "./graph/check.js";
import { ask, type AskResult } from "./ask/ask.js";

export { CODE_EXTENSIONS };
export type { BuildResult, BuildProgress, CheckResult, GraphBuildResult, GraphCheckResult, AskResult };

export interface InitOptions {
  /** Code extensions to include. Default: {@link CODE_EXTENSIONS}. */
  extensions?: string[];
  /** Repo-relative directory prefixes to limit the concept pass (`--only-dir`). */
  onlyDirs?: string[];
  /** Progress callback for long builds. */
  onProgress?: (info: BuildProgress) => void;
}

export interface CheckRunOptions {
  extensions?: string[];
}

export interface GraphRunOptions {
  /** Run the Tier-2 LLM meaning pass (summary + crux). Absent → Tier-1 only. */
  llm?: boolean;
  /** Max files summarized in parallel during the LLM pass. */
  concurrency?: number;
  /** Replay unchanged files from the extraction cache (default true). */
  reuse?: boolean;
  /** Opt-in compiler-grade LSP edge enrichment (`graft build --lsp`). */
  lsp?: boolean;
  /** Repo-relative directory prefixes to limit the build to (`--only-dir`). */
  onlyDirs?: string[];
  onProgress?: GraphBuildOptions["onProgress"];
}

export class Graft {
  private cfg: ResolvedConfig;
  private readonly userConfig: EngineConfig;
  /** Tokens and calls spent through this engine so far (`build --deep` prints it). */
  readonly usage: UsageTotals = emptyUsage();

  constructor(config: EngineConfig = {}) {
    this.userConfig = config;
    this.cfg = resolveConfig(config);
  }

  /** Build the `.context/` graph from the repo at `dir`. */
  async init(dir: string, opts: InitOptions = {}): Promise<BuildResult> {
    return buildContext(dir, {
      contextDir: this.cfg.contextDir,
      extensions: opts.extensions,
      onlyDirs: opts.onlyDirs,
      model: this.modelLabel(),
      summarizer: this.summarizer(),
      synthesizer: this.synthesizer(),
      onProgress: opts.onProgress,
    });
  }

  /** Report whether the committed `.context/` markdown graph is in sync with the code. */
  check(dir: string, opts: CheckRunOptions = {}): CheckResult {
    return checkContext(dir, { contextDir: this.cfg.contextDir, extensions: opts.extensions });
  }

  /** Report whether the committed `graph.json` is in sync with the code (Tier-1 diff).
   * Async because the breadth tier warms WASM grammars before re-extraction. */
  checkGraph(dir: string): Promise<GraphCheckResult> {
    return checkGraph(dir, { contextDir: this.cfg.contextDir });
  }

  /**
   * Build `.context/graph.json` — a per-symbol code graph from tree-sitter.
   * Tier-1 (structure) always runs; the Tier-2 meaning layer runs only when
   * `opts.llm` is set. Either way the prior meaning layer is preserved.
   */
  graph(dir: string, opts: GraphRunOptions = {}): Promise<GraphBuildResult> {
    return buildGraph(dir, {
      contextDir: this.cfg.contextDir,
      summarizer: opts.llm ? this.cruxSummarizer() : undefined,
      concurrency: opts.concurrency,
      reuse: opts.reuse,
      lsp: opts.lsp,
      onlyDirs: opts.onlyDirs,
      onProgress: opts.onProgress,
    });
  }

  /**
   * Answer a plain-words query from the committed `graft/` graph — the active
   * channel. Deterministic and $0: routes structural queries to the wiring
   * edges and everything else to a lexical rank over concepts + symbols.
   */
  ask(dir: string, query: string, opts: { limit?: number; source?: boolean; full?: boolean; in?: string; graphRank?: boolean } = {}): AskResult {
    return ask(dir, query, {
      contextDir: this.cfg.contextDir,
      limit: opts.limit,
      source: opts.source,
      full: opts.full,
      in: opts.in,
      graphRank: opts.graphRank,
    });
  }

  private _chatModel?: ChatModel;
  private readonly _passModels = new Map<LlmPass, ChatModel>();

  /** The shared transport, or a clear error telling the user how to set a key. */
  private chatModel(): ChatModel {
    if (this._chatModel) return this._chatModel;
    this._chatModel = this.cfg.chatModel ? meter(this.cfg.chatModel, this.usage) : this.buildModel(this.cfg);
    return this._chatModel;
  }

  /** The transport for one pass: the shared model unless a `GRAFT_<PASS>_*` variable
   * singles this pass out, in which case its own resolved config builds a second
   * client. Every model meters into the same `usage` total. An injected `chatModel`
   * always wins — a programmatic caller has chosen the transport for everything. */
  private chatModelFor(pass: LlmPass): ChatModel {
    if (this.cfg.chatModel || !hasPassOverrides(pass)) return this.chatModel();
    let m = this._passModels.get(pass);
    if (!m) {
      m = this.buildModel(resolveConfig(this.userConfig, pass));
      this._passModels.set(pass, m);
    }
    return m;
  }

  private buildModel(cfg: ResolvedConfig): ChatModel {
    if (!cfg.apiKey) {
      throw new Error(
        "No API key. Set GRAFT_API_KEY (and GRAFT_PROVIDER / GRAFT_BASE_URL / GRAFT_MODEL " +
          "for your provider) to build or summarize the graph.",
      );
    }
    return meter(
      createChatModel({ provider: cfg.provider, apiKey: cfg.apiKey, model: cfg.model, baseUrl: cfg.baseUrl, headers: cfg.headers }),
      this.usage,
    );
  }

  private synthesizer(): Synthesizer {
    return this.cfg.synthesizer ?? new ChatSynthesizer(this.chatModelFor("synth"));
  }

  /** Per-node crux summarizer for the code graph's Tier-2 pass. */
  private cruxSummarizer(): CruxSummarizer {
    return this.cfg.cruxSummarizer ?? new ChatCruxSummarizer(this.chatModelFor("crux"));
  }

  private summarizer(): Summarizer {
    return this.cfg.summarizer ?? new ChatSummarizer(this.chatModelFor("summary"));
  }

  /** `provider:model` per pass — identical labels collapse to one. */
  passLabels(): Record<LlmPass, string> {
    const out = {} as Record<LlmPass, string>;
    for (const pass of LLM_PASSES) {
      const cfg = this.cfg.chatModel || !hasPassOverrides(pass) ? this.cfg : resolveConfig(this.userConfig, pass);
      out[pass] = this.cfg.chatModel ? this.cfg.chatModel.label : `${cfg.provider}:${cfg.model}`;
    }
    return out;
  }

  /** Human label for the active model(s), recorded in the manifest: one label when
   * every pass shares a model, else `summary=…; synth=…; crux=…`. */
  private modelLabel(): string {
    if (this.cfg.chatModel) return this.cfg.chatModel.label;
    if (this.cfg.synthesizer || this.cfg.summarizer || this.cfg.cruxSummarizer) return "custom";
    const labels = this.passLabels();
    const distinct = new Set(Object.values(labels));
    if (distinct.size === 1) return labels.summary;
    return LLM_PASSES.map((p) => `${p}=${labels[p]}`).join("; ");
  }
}
