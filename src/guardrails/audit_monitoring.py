"""Audit logging and monitoring helpers for the defense pipeline."""
from __future__ import annotations
import json
import time
from datetime import datetime
from typing import Any

from google.adk.plugins import base_plugin
from google.genai import types


class AuditLogPlugin(base_plugin.BasePlugin):
    """Log every request/output pair along with blocking reasons."""

    instance: "AuditLogPlugin" | None = None

    def __init__(self, filepath: str = "audit_log.json") -> None:
        super().__init__(name="audit_log")
        self.filepath = filepath
        self.logs: list[dict[str, Any]] = []
        self._pending_stack: list[dict[str, Any]] = []
        AuditLogPlugin.instance = self

    def _extract_text(self, content: types.Content | None) -> str:
        text = ""
        if content and content.parts:
            for part in content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    def mark_blocked(self, layer: str, reason: str | None = None) -> None:
        """Annotate the current interaction with a blocking layer and reason."""
        if not self._pending_stack:
            return
        entry = self._pending_stack[-1]
        entry["blocked_layer"] = layer
        if reason:
            existing = entry.get("blocked_reason")
            entry["blocked_reason"] = (
                f"{existing}; {reason}" if existing else reason
            )

    async def on_user_message_callback(self, *, invocation_context, user_message: types.Content):
        """Start a fresh log entry for the incoming user request."""
        user_id = (
            invocation_context.user_id
            if invocation_context and hasattr(invocation_context, "user_id")
            else "anonymous"
        )
        text = self._extract_text(user_message)
        entry = {
            "request_id": len(self.logs) + 1,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "input_text": text,
            "blocked_layer": None,
            "blocked_reason": None,
            "metadata": {},
            "response_text": None,
            "latency_ms": None,
            "_start_perf": time.perf_counter(),
        }
        self.logs.append(entry)
        self._pending_stack.append(entry)
        return None

    async def after_model_callback(self, *, callback_context, llm_response):
        """Capture the LLM output and latency once processing finishes."""
        if not self._pending_stack:
            return llm_response

        entry = self._pending_stack.pop()
        output_text = self._extract_text(getattr(llm_response, "content", None))
        start_perf = entry.pop("_start_perf", None)
        if start_perf is not None:
            entry["latency_ms"] = (time.perf_counter() - start_perf) * 1000
        entry["response_text"] = output_text
        entry["timestamp"] = datetime.utcnow().isoformat()
        return llm_response

    def export_json(self, filepath: str | None = None) -> None:
        """Write the accumulated audit log to disk."""
        target = filepath or self.filepath
        serializable = [
            {k: v for k, v in entry.items() if not k.startswith("_")}
            for entry in self.logs
        ]
        with open(target, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)


class MonitoringAlert:
    """Track high-level metrics across guardrail plugins and raise alerts."""

    def __init__(
        self,
        input_plugin: Any = None,
        output_plugin: Any = None,
        rate_limit_threshold: float = 0.3,
        judge_fail_threshold: float = 0.25,
        redaction_threshold: float = 0.35,
    ) -> None:
        self.input_plugin = input_plugin
        self.output_plugin = output_plugin
        self.rate_limit_threshold = rate_limit_threshold
        self.judge_fail_threshold = judge_fail_threshold
        self.redaction_threshold = redaction_threshold

    def check_metrics(self) -> dict[str, float]:
        """Compute rates and print alerts if thresholds are exceeded."""
        rate_limit_rate = (
            (self.input_plugin.rate_limit_blocks / self.input_plugin.total_count)
            if self.input_plugin and self.input_plugin.total_count
            else 0.0
        )
        judge_fail_rate = (
            (self.output_plugin.blocked_count / self.output_plugin.total_count)
            if self.output_plugin and self.output_plugin.total_count
            else 0.0
        )
        redaction_rate = (
            (self.output_plugin.redacted_count / self.output_plugin.total_count)
            if self.output_plugin and self.output_plugin.total_count
            else 0.0
        )

        print("\n--- Monitoring & Alerts ---")
        print(f"Rate-limit hit rate: {rate_limit_rate:.1%}")
        print(f"Judge block rate:    {judge_fail_rate:.1%}")
        print(f"Redaction rate:      {redaction_rate:.1%}")

        alerts = []
        if rate_limit_rate > self.rate_limit_threshold:
            alerts.append("High rate-limit hit rate — investigate abuse or adjust limits.")
        if judge_fail_rate > self.judge_fail_threshold:
            alerts.append("Judge is blocking many responses — check false positives.")
        if redaction_rate > self.redaction_threshold:
            alerts.append("Redactions are frequent — verify your PII filtering rules.")

        for alert in alerts:
            print(f"ALERT: {alert}")

        return {
            "rate_limit_rate": rate_limit_rate,
            "judge_fail_rate": judge_fail_rate,
            "redaction_rate": redaction_rate,
            "alerts": alerts,
        }
