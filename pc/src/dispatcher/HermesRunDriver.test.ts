import { strict as assert } from "node:assert";
import test from "node:test";
import {
  eventDelta,
  looksLikeToolPayload,
  MAX_VISIBLE_DELTA_CHARS,
  VisibleXmlStreamFilter
} from "./HermesRunDriver.js";

function collect(chunks: string[]): string {
  const filter = new VisibleXmlStreamFilter();
  return chunks.map((chunk) => filter.push(chunk)).join("");
}

test("VisibleXmlStreamFilter strips streamed invoke XML without leaking partial tags", () => {
  const visible = collect([
    "I'll check that.\n<in",
    "voke name=\"read_file\">",
    "{\"path\":\"/tmp/x\"}",
    "</invoke>\nDone."
  ]);

  assert.equal(visible, "I'll check that.\n\nDone.");
  assert.equal(visible.includes("<invoke"), false);
  assert.equal(visible.includes("read_file"), false);
});

test("VisibleXmlStreamFilter strips streamed tool_call XML and preserves later text", () => {
  const visible = collect([
    "Before ",
    "<tool_call>{\"name\":\"terminal\"}",
    "</tool_call>",
    " after"
  ]);

  assert.equal(visible, "Before  after");
  assert.equal(visible.includes("tool_call"), false);
  assert.equal(visible.includes("terminal"), false);
});

test("VisibleXmlStreamFilter preserves normal prose", () => {
  const visible = collect(["Use ", "normal text ", "here."]);
  assert.equal(visible, "Use normal text here.");
});

test("eventDelta only exposes assistant text delta events", () => {
  assert.equal(eventDelta({
    event: "message",
    data: { event: "message.delta", delta: "visible" },
    raw: ""
  }), "visible");
  assert.equal(eventDelta({
    event: "response.output_text.delta",
    data: { type: "response.output_text.delta", text: "also visible" },
    raw: ""
  }), "also visible");
});

test("eventDelta hides tool and reasoning payload text from visible chat", () => {
  const phoneDump = JSON.stringify({
    event: "tool.completed",
    tool: "phone_observe",
    output: { success: true, nodes: [{ text: "huge screen dump" }] }
  });

  assert.equal(eventDelta({
    event: "message",
    data: { event: "tool.completed", tool: "phone_observe", text: phoneDump },
    raw: ""
  }), undefined);
  assert.equal(eventDelta({
    event: "message",
    data: { event: "reasoning.available", text: phoneDump },
    raw: ""
  }), undefined);
});

test("eventDelta drops oversized deltas that are tool payloads riding the text channel", () => {
  const hugeScreenDump = JSON.stringify({
    success: true,
    nodes: Array.from({ length: 200 }, (_, i) => ({
      id: `node-${i}`,
      text: `Some accessibility node label ${i} with content`,
      bounds: { x: i * 10, y: i * 20, width: 100, height: 40 },
      contentDescription: `desc ${i}`
    }))
  });
  assert.ok(hugeScreenDump.length > MAX_VISIBLE_DELTA_CHARS, "fixture must exceed cap");

  assert.equal(eventDelta({
    event: "message",
    data: { event: "message.delta", delta: hugeScreenDump },
    raw: ""
  }), undefined);
});

test("eventDelta drops small but shaped tool payloads that ride message.delta", () => {
  const smallToolPayload = JSON.stringify({
    tool_call_id: "call_abc123",
    name: "phone_observe",
    content: { nodes: [] }
  });
  assert.ok(smallToolPayload.length < MAX_VISIBLE_DELTA_CHARS, "fixture must be under cap");

  assert.equal(eventDelta({
    event: "message",
    data: { event: "message.delta", delta: smallToolPayload },
    raw: ""
  }), undefined);
});

test("eventDelta still passes ordinary short text deltas after the shape check", () => {
  assert.equal(eventDelta({
    event: "message",
    data: { event: "message.delta", delta: "I'll check that." },
    raw: ""
  }), "I'll check that.");
  assert.equal(eventDelta({
    event: "message",
    data: { event: "assistant.delta", delta: "Sure, opening Settings now." },
    raw: ""
  }), "Sure, opening Settings now.");
});

test("looksLikeToolPayload identifies JSON tool outputs by key signature", () => {
  assert.equal(looksLikeToolPayload(""), false);
  assert.equal(looksLikeToolPayload("Just plain text."), false);
  // Distinctive keys alone are enough.
  assert.equal(looksLikeToolPayload("{\"tool_call_id\":\"abc\"}"), true);
  assert.equal(looksLikeToolPayload("{\"nodes\":[]}"), true);
  assert.equal(looksLikeToolPayload("{\"bounds\":{}}"), true);
  assert.equal(looksLikeToolPayload("  {\"contentDescription\":\"foo\"}"), true);
  // `name` alone is too generic to count.
  assert.equal(looksLikeToolPayload("{\"name\":\"phone_observe\"}"), false);
  // `name` + structured content is the tool-call shape.
  assert.equal(looksLikeToolPayload("{\"name\":\"phone_observe\",\"content\":{}}"), true);
  assert.equal(looksLikeToolPayload("{\"name\":\"read_file\",\"arguments\":{\"path\":\"/tmp\"}}"), true);
  assert.equal(looksLikeToolPayload("{\"name\":\"x\",\"input\":{\"y\":1}}"), true);
  // Non-JSON or non-object shapes never match.
  assert.equal(looksLikeToolPayload("{not valid json"), false);
  assert.equal(looksLikeToolPayload("[]"), false);
  assert.equal(looksLikeToolPayload("\"plain json string\""), false);
});
