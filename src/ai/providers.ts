import type { Summarizer } from "./summarize.js";
import type { Synthesizer } from "./synthesize.js";
import type { CruxSummarizer } from "./crux.js";
import type { ChatModel } from "./llm/types.js";
import type { ProviderKind } from "./llm/factory.js";

/**
 * User-facing configuration. Anything omitted falls back to environment
 * variables and then to sensible defaults.
 *
 * graft is vendor-neutral: `provider` names only the WIRE FORMAT, not a company.
 * `openai` speaks the OpenAI-compatible API — point `baseUrl` at OpenRouter,
 * Fireworks, a LiteLLM proxy, Groq, a local server, or OpenAI itself, and pass
 * your own key. `anthropic` speaks the native Messages API. Any LLM-backed
 * operation needs an API key.
 */
export interface EngineConfig {
  /** Where the graph lives. Env: GRAFT_DIR. Default: `<repo>/.context`. */
  contextDir?: string;

  /** Wire format / SDK. Env: GRAFT_PROVIDER. Default: `openai`. */
  provider?: ProviderKind;
  /** API key for the chosen provider. Env: GRAFT_API_KEY (legacy: OPENROUTER_API_KEY). */
  apiKey?: string;
  /** Model id. Env: GRAFT_MODEL. Provider-specific default. */
  model?: string;
  /** Base URL for OpenAI-compatible endpoints. Env: GRAFT_BASE_URL. */
  baseUrl?: string;
  /** Extra headers on every LLM call — a gateway's project/team tag (Bedrock's
   * `anthropic-workspace-id`, OpenRouter's `X-Title`, …). Env: GRAFT_LLM_HEADERS
   * as `name=value, name=value`. */
  headers?: Record<string, string>;

  // --- advanced: bring your own components ---
  /** Override the whole transport (skips provider/apiKey/baseUrl). */
  chatModel?: ChatModel;
  /** Override the synthesizer. */
  synthesizer?: Synthesizer;
  /** Override the code summarizer. */
  summarizer?: Summarizer;
  /** Override the per-symbol crux summarizer. */
  cruxSummarizer?: CruxSummarizer;
}

/** Fully-resolved configuration with all defaults applied. */
export interface ResolvedConfig {
  contextDir?: string;
  provider: ProviderKind;
  apiKey?: string;
  model: string;
  baseUrl?: string;
  headers?: Record<string, string>;
  /** True when the key came from the deprecated OPENROUTER_* fallback. */
  usedLegacyEnv: boolean;
  chatModel?: ChatModel;
  synthesizer?: Synthesizer;
  summarizer?: Summarizer;
  cruxSummarizer?: CruxSummarizer;
}

const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";
const ORCAROUTER_BASE_URL = "https://api.orcarouter.ai/v1";

/** Per-provider default model. */
export const DEFAULT_MODELS: Record<ProviderKind, string> = {
  openai: "openai/gpt-4o-mini",
  anthropic: "claude-sonnet-5",
  // Provider-prefixed so the LiteLLM proxy routes it; override with GRAFT_MODEL.
  litellm: "openai/gpt-4o-mini",
  // Provider-prefixed so the OrcaRouter gateway routes it; override with GRAFT_MODEL.
  orcarouter: "openai/gpt-4o-mini",
};

export const DEFAULTS = {
  provider: "openai" as ProviderKind,
  model: DEFAULT_MODELS.openai,
} as const;

/** `name=value, name=value` → headers. Blank entries and entries without `=` are dropped. */
export function parseHeaderList(raw: string | undefined): Record<string, string> {
  const out: Record<string, string> = {};
  for (const part of (raw ?? "").split(",")) {
    const eq = part.indexOf("=");
    if (eq <= 0) continue;
    const name = part.slice(0, eq).trim();
    const value = part.slice(eq + 1).trim();
    if (name && value) out[name] = value;
  }
  return out;
}

/** The three LLM passes of a `--deep` build, each overridable on its own. */
export type LlmPass = "summary" | "synth" | "crux";
export const LLM_PASSES: readonly LlmPass[] = ["summary", "synth", "crux"];
const PASS_SETTINGS = ["PROVIDER", "MODEL", "BASE_URL", "API_KEY", "LLM_HEADERS"] as const;

/** `GRAFT_<PASS>_<SETTING>` — e.g. `GRAFT_SYNTH_MODEL` — or undefined when no pass is given. */
function passEnv(pass: LlmPass | undefined, setting: (typeof PASS_SETTINGS)[number]): string | undefined {
  return pass ? process.env[`GRAFT_${pass.toUpperCase()}_${setting}`] : undefined;
}

/** True when any `GRAFT_<PASS>_*` variable is set for this pass. */
export function hasPassOverrides(pass: LlmPass): boolean {
  return PASS_SETTINGS.some((s) => passEnv(pass, s) !== undefined);
}

/**
 * Merge user config with environment variables and defaults. With `pass`, each
 * setting first consults its `GRAFT_<PASS>_*` variable — a pass-specific value
 * beats a CLI flag, which beats the global env, which beats the default. Each
 * setting falls back independently: a pass that switches provider must also name
 * its model (and, off the default endpoint, its base URL and headers).
 */
export function resolveConfig(config: EngineConfig = {}, pass?: LlmPass): ResolvedConfig {
  const env = process.env;
  const provider =
    (passEnv(pass, "PROVIDER") as ProviderKind | undefined) ??
    config.provider ??
    (env.GRAFT_PROVIDER as ProviderKind | undefined) ??
    DEFAULTS.provider;

  const explicitKey = passEnv(pass, "API_KEY") ?? config.apiKey ?? env.GRAFT_API_KEY;
  const legacyKey = env.OPENROUTER_API_KEY;
  const apiKey = explicitKey ?? legacyKey ?? env.ORCAROUTER_API_KEY;
  const usedLegacyEnv = !explicitKey && !!legacyKey;

  const model =
    passEnv(pass, "MODEL") ??
    config.model ??
    env.GRAFT_MODEL ??
    env.GRAFT_OPENROUTER_MODEL ??
    env.ORCAROUTER_MODEL ??
    DEFAULT_MODELS[provider];

  let baseUrl =
    passEnv(pass, "BASE_URL") ?? config.baseUrl ?? env.GRAFT_BASE_URL ?? env.OPENROUTER_BASE_URL ?? env.ORCAROUTER_BASE_URL;
  // Back-compat: an existing setup with only OPENROUTER_API_KEY keeps hitting
  // OpenRouter without any config change.
  if (!baseUrl && provider === "openai" && usedLegacyEnv) baseUrl = OPENROUTER_BASE_URL;
  // The orcarouter provider points at the gateway unless a base URL is given.
  if (!baseUrl && provider === "orcarouter") baseUrl = ORCAROUTER_BASE_URL;

  const merged = {
    ...(provider === "openai" && baseUrl?.includes("openrouter.ai") ? { "X-Title": "graft" } : {}),
    ...(passEnv(pass, "LLM_HEADERS") !== undefined
      ? parseHeaderList(passEnv(pass, "LLM_HEADERS"))
      : (config.headers ?? parseHeaderList(env.GRAFT_LLM_HEADERS))),
  };
  const headers = Object.keys(merged).length ? merged : undefined;

  return {
    contextDir: config.contextDir ?? env.GRAFT_DIR,
    provider,
    apiKey,
    model,
    baseUrl,
    headers,
    usedLegacyEnv,
    chatModel: config.chatModel,
    synthesizer: config.synthesizer,
    summarizer: config.summarizer,
    cruxSummarizer: config.cruxSummarizer,
  };
}
