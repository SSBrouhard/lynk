import type { ChatAttachment } from "../protocol/messages.js";
import type { HermesRunStatus, HermesRunTransport, HermesSseEvent } from "./HermesApiClient.js";

export interface HermesActiveRun {
  runId: string;
  sessionId: string;
  controller: AbortController;
}

export type HermesRunDriverEvent =
  | { type: "delta"; delta: string; accumulated: string; raw: HermesSseEvent }
  | { type: "tool"; toolName?: string; status?: string; raw: HermesSseEvent }
  | { type: "status"; status: string; raw: HermesSseEvent };

export interface HermesRunDriverResult {
  active: HermesActiveRun;
  status: HermesRunStatus;
  finalText: string;
}

export class HermesRunDriver {
  constructor(
    private readonly api: HermesRunTransport,
    private readonly runTimeoutMs: number
  ) {}

  async createRun(options: {
    input: string;
    sessionId: string;
    model?: string;
    instructions?: string;
    idempotencyKey?: string;
    attachments?: ChatAttachment[];
    serviceTier?: "priority" | null;
  }): Promise<HermesActiveRun> {
    const created = await this.api.createRun(options);
    return {
      runId: created.runId,
      sessionId: created.sessionId ?? options.sessionId,
      controller: new AbortController()
    };
  }

  async streamRun(
    active: HermesActiveRun,
    onEvent: (event: HermesRunDriverEvent) => void
  ): Promise<HermesRunDriverResult> {
    let latestOutput = "";
    const visibleFilter = new VisibleXmlStreamFilter();
    await this.withTimeout(
      this.api.streamRunEvents(active.runId, (event) => {
        latestOutput = this.handleEvent(event, latestOutput, visibleFilter, onEvent);
      }, active.controller.signal),
      this.runTimeoutMs
    );

    const status = await this.api.getRun(active.runId);
    const error = outputText(status.error);
    if (error) {
      throw new Error(error);
    }
    return {
      active,
      status,
      finalText: sanitizeVisibleXml(outputText(status.output) ?? latestOutput)
    };
  }

  async stopRun(active: HermesActiveRun): Promise<void> {
    active.controller.abort();
    await this.api.stopRun(active.runId);
  }

  async steerRun(active: HermesActiveRun, text: string, attachments?: ChatAttachment[], serviceTier?: "priority" | null): Promise<void> {
    await this.api.createRun({
      input: `Additional user guidance for the active Hermes task:\n${text.trim()}`,
      sessionId: active.sessionId,
      idempotencyKey: `hermes-steer-${active.runId}-${Date.now()}`,
      attachments,
      serviceTier
    });
  }

  private handleEvent(
    event: HermesSseEvent,
    latestOutput: string,
    visibleFilter: VisibleXmlStreamFilter,
    onEvent: (event: HermesRunDriverEvent) => void
  ): string {
    const toolName = eventToolName(event);
    const status = eventStatus(event);
    if (toolName || event.event.toLowerCase().includes("tool")) {
      onEvent({ type: "tool", toolName, status, raw: event });
    } else if (status && !["completed", "failed", "cancelled"].includes(status)) {
      onEvent({ type: "status", status, raw: event });
    }

    const delta = eventDelta(event);
    if (!delta) {
      return latestOutput;
    }
    const visibleDelta = visibleFilter.push(delta);
    if (!visibleDelta) {
      return latestOutput;
    }
    const accumulated = latestOutput + visibleDelta;
    onEvent({ type: "delta", delta: visibleDelta, accumulated, raw: event });
    return accumulated;
  }

  private async withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
    let timer: NodeJS.Timeout | undefined;
    try {
      return await Promise.race([
        promise,
        new Promise<T>((_, reject) => {
          timer = setTimeout(() => reject(new Error(`Hermes task timed out after ${Math.round(timeoutMs / 1000)} seconds`)), timeoutMs);
        })
      ]);
    } finally {
      if (timer) {
        clearTimeout(timer);
      }
    }
  }
}

const XML_BLOCK_TAGS = [
  "think",
  "thinking",
  "reasoning",
  "thought",
  "reasoning_scratchpad",
  "tool_call",
  "tool_calls",
  "tool_result",
  "function_call",
  "function_calls",
  "invoke",
  "invoke_tool"
];

const XML_BLOCK_TAG_PATTERN = XML_BLOCK_TAGS.join("|");

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function startsProtectedXmlTag(fragment: string): boolean {
  const lower = fragment.toLowerCase();
  return XML_BLOCK_TAGS.some((tag) => `<${tag}`.startsWith(lower) || lower.startsWith(`<${tag}`))
    || "<function".startsWith(lower)
    || lower.startsWith("<function");
}

function sanitizeVisibleXml(text: string): string {
  if (!text) {
    return "";
  }
  let result = text;
  for (const tag of XML_BLOCK_TAGS) {
    const escaped = escapeRegExp(tag);
    result = result.replace(new RegExp(`<${escaped}\\b[^>]*>[\\s\\S]*?<\\/${escaped}>`, "gi"), "");
  }
  result = result.replace(
    /(^|[\n\r.!?:])([ \t]*)<function\b[^>]*\bname\s*=[^>]*>[\s\S]*?<\/function>/gi,
    "$1"
  );
  result = result.replace(
    new RegExp(`<(?:${XML_BLOCK_TAG_PATTERN})\\b[^>]*>[\\s\\S]*$`, "i"),
    ""
  );
  result = result.replace(
    /(^|[\n\r.!?:])([ \t]*)<function\b[^>]*\bname\s*=[^>]*>[\s\S]*$/i,
    "$1"
  );
  result = result.replace(new RegExp(`<\\/(?:${XML_BLOCK_TAG_PATTERN}|function)>\\s*`, "gi"), "");
  return result;
}

export class VisibleXmlStreamFilter {
  private raw = "";
  private emitted = "";

  push(delta: string): string {
    this.raw += delta;
    let visible = sanitizeVisibleXml(this.raw);

    // Hold back a partial protected tag at the tail so the UI does not briefly
    // render fragments like "<invoke" before the closing ">" arrives.
    const lastLt = visible.lastIndexOf("<");
    if (lastLt >= 0) {
      const tail = visible.slice(lastLt);
      if (!tail.includes(">") && startsProtectedXmlTag(tail)) {
        visible = visible.slice(0, lastLt);
      }
    }

    if (visible.length < this.emitted.length) {
      this.emitted = visible;
      return "";
    }
    const next = visible.slice(this.emitted.length);
    this.emitted = visible;
    return next;
  }
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" ? value as Record<string, unknown> : undefined;
}

function firstStringField(value: unknown, keys: string[]): string | undefined {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      const nested = firstStringField(item, keys);
      if (nested) {
        return nested;
      }
    }
    return undefined;
  }
  const record = asRecord(value);
  if (!record) {
    return undefined;
  }
  for (const key of keys) {
    const field = record[key];
    if (typeof field === "string" && field.trim()) {
      return field.trim();
    }
  }
  for (const field of Object.values(record)) {
    const nested = firstStringField(field, keys);
    if (nested) {
      return nested;
    }
  }
  return undefined;
}

function outputText(value: unknown): string | undefined {
  return firstStringField(value, ["output", "final_output", "finalMessage", "message", "text", "content", "delta"]);
}

function eventStatus(event: HermesSseEvent): string | undefined {
  const record = asRecord(event.data);
  const value = record?.status ?? record?.state ?? record?.phase;
  return typeof value === "string" ? value.toLowerCase() : undefined;
}

function eventToolName(event: HermesSseEvent): string | undefined {
  const record = asRecord(event.data);
  const nested = asRecord(record?.data) ?? record;
  const value = nested?.toolName ?? nested?.tool ?? nested?.name ?? nested?.function_name;
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

const VISIBLE_DELTA_EVENT_NAMES = new Set([
  "assistant.delta",
  "message.delta",
  "response.output_text.delta"
]);

function eventNames(event: HermesSseEvent): string[] {
  const record = asRecord(event.data);
  const nested = asRecord(record?.data);
  return [
    event.event,
    typeof record?.event === "string" ? record.event : undefined,
    typeof record?.type === "string" ? record.type : undefined,
    typeof nested?.event === "string" ? nested.event : undefined,
    typeof nested?.type === "string" ? nested.type : undefined
  ]
    .filter((value): value is string => Boolean(value && value.trim()))
    .map((value) => value.trim().toLowerCase());
}

function isVisibleDeltaEvent(event: HermesSseEvent): boolean {
  const names = eventNames(event);
  return names.some((name) => VISIBLE_DELTA_EVENT_NAMES.has(name));
}

// Size cap: legitimate assistant text deltas are tiny. Anything past this is
// almost certainly a tool-result payload (e.g. phone_observe screen dumps)
// riding the delta channel.
export const MAX_VISIBLE_DELTA_CHARS = 2048;

// Heuristic shape match for JSON-encoded tool outputs that masquerade as a
// text delta. Phone MCP observations, tool_use blocks, and most structured
// payloads all carry one of these markers.
const TOOL_PAYLOAD_KEYS = [
  "tool_call_id",
  "tool_use_id",
  "tool_calls",
  "tool_result",
  "function_call",
  "is_error",
  "content_description",
  "contentDescription",
  "nodes",
  "bounds",
  "accessibility_tree"
] as const;

export function looksLikeToolPayload(value: string): boolean {
  if (!value) {
    return false;
  }
  const trimmed = value.trimStart();
  if (trimmed.length === 0 || trimmed[0] !== "{") {
    return false;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return false;
  }
  if (!parsed || typeof parsed !== "object") {
    return false;
  }
  const record = parsed as Record<string, unknown>;
  // Distinctive keys: any one of these is enough.
  if (TOOL_PAYLOAD_KEYS.some((key) => key in record)) {
    return true;
  }
  // `name` alone is too generic, but `name` together with structured content
  // (object/array `content`, `input`, or `arguments`) is a tool-call shape.
  if (typeof record.name === "string" && record.name.length > 0) {
    if (
      record.content !== undefined && (typeof record.content === "object") ||
      record.input !== undefined ||
      record.arguments !== undefined
    ) {
      return true;
    }
  }
  return false;
}

export function eventDelta(event: HermesSseEvent): string | undefined {
  if (!isVisibleDeltaEvent(event)) {
    return undefined;
  }
  const record = asRecord(event.data);
  const nested = asRecord(record?.data) ?? record;
  for (const source of [nested, record]) {
    for (const key of ["delta", "text_delta", "output_text", "text"]) {
      const value = source?.[key];
      if (typeof value !== "string" || value.length === 0) {
        continue;
      }
      if (value.length > MAX_VISIBLE_DELTA_CHARS) {
        return undefined;
      }
      if (looksLikeToolPayload(value)) {
        return undefined;
      }
      return value;
    }
  }
  return undefined;
}
