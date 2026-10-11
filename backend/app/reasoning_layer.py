"""
reasoning_layer.py
------------------
AI reasoning layer for Pulse RAG.

Responsibilities:
  - Accept a structured, classified lab report (dict).
  - Retrieve relevant medical context for abnormal parameters (FAISS RAG).
  - Build a focused prompt and call the Groq LLM once.
  - Return a strictly structured, patient-friendly response.

Output schema:
    {
      "summary":             str,        # short key points, specific values
      "intellectual_audio":  str,        # plain paragraph meant to be spoken
      "preventive_guidance": str,        # lifestyle-only, non-diagnostic
      "doctor_questions":    [str, str, str]   # exactly 3
    }

Privacy: the patient's name is deliberately NOT sent to the LLM. It is not
needed to explain lab values.

Usage:
    from app.reasoning_layer import LabReportReasoningAgent

    agent = LabReportReasoningAgent()
    result = agent.analyze(classified_report, language="English")
"""

import json
import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import get_settings
from app.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

_FALLBACK_QUESTIONS = [
    "Which of my results require attention first?",
    "What lifestyle changes do you recommend based on my results?",
    "When should I repeat these tests?",
]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a compassionate, patient-facing health education assistant.
Your job is to help patients understand their lab report results — NOT to diagnose them.

STRICT RULES:
1. Use cautious language: "may suggest", "could indicate", "sometimes associated with".
2. NEVER diagnose a condition. NEVER name a specific disease as a confirmed finding.
3. NEVER hallucinate — only comment on parameters explicitly provided.
4. Mention each abnormal parameter by name AND its actual measured value.
5. Preventive guidance must be lifestyle-only: diet, exercise, hydration, sleep, stress.
6. Do NOT recommend specific medications or supplements by brand name.
7. Be empathetic, clear, and reassuring in tone.
8. The "summary" field MUST be a single string formatted with `\\n` newline characters to create a list of 6-7 SHORT and SIMPLE key points.
   - Keep each bullet point to a single, concise sentence.
   - Avoid long, complex medical explanations.
   - Start each point with a bolded category like "**Overview:** ".
9. The "intellectual_audio" field MUST be a beautiful, highly intelligent, continuous and compassionate conversational paragraph explaining the summary points without ANY list markers, bullet points, numbers, or markdown whatsoever. It is designed to be spoken aloud.
10. Return ONLY a valid JSON object — no extra text.
"""

_USER_PROMPT_TEMPLATE = """\
=== REPORT INFORMATION ===
Report Date     : {report_date}
Target Language : {language}

=== FULL LAB RESULTS ===
{results_table}

=== ABNORMAL PARAMETERS (flagged for your focus) ===
{abnormal_summary}

=== RELEVANT MEDICAL CONTEXT (retrieved from knowledge base) ===
{rag_context}

=== YOUR TASK ===
Write a response for the patient strictly in {language} using ONLY the JSON structure below.
Do not add any text before or after the JSON.

{{
  "summary": "- **Overview:** <Short, simple 1-sentence point>\\n- **Key Concerns:** <Short, simple 1-sentence point>\\n- **Next Steps:** <Short, simple 1-sentence point>\\n- **<Category>:** <Short point>\\n- **<Category>:** <Short point>\\n- **<Category>:** <Short point>",
  "intellectual_audio": "<A warm, intellectual conversational paragraph explaining the results designed strictly for an audible voice-over. NO list markers or markdown.>",
  "preventive_guidance": "<Single paragraph of 3 to 5 specific, actionable lifestyle tips directly relevant to the flagged parameters. No diagnoses. No medications.>",
  "doctor_questions": [
    "<Targeted question about the most concerning abnormal value>",
    "<Question about lifestyle or diet changes specific to the patient's flagged results>",
    "<Question about follow-up tests or monitoring timeline>"
  ]
}}
"""


# ---------------------------------------------------------------------------
# Helper: Format results as readable table
# ---------------------------------------------------------------------------

def _build_results_table(parameters: list[dict]) -> str:
    """
    Convert parameters list into a plain-text aligned table.

    Example:
        ⚠️  Hemoglobin            9.8  g/dL        | Ref: 12.0–15.5  | Status: Low
        ✅  TSH                   3.2  mIU/L        | Ref: 0.5–4.5    | Status: Normal
    """
    lines: list[str] = []
    for p in parameters:
        flag  = "⚠️ " if p.get("status") in ("High", "Low") else "✅ "
        name  = p.get("name", "Unknown")
        value = p.get("value", "N/A")
        unit  = p.get("unit", "")
        rlo   = p.get("reference_low", "?")
        rhi   = p.get("reference_high", "?")
        status = p.get("status", "Unknown")
        lines.append(
            f"{flag}{name:<28} {value!s:<8} {unit:<12} | Ref: {rlo}–{rhi:<10} | {status}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: One-line abnormal summary injected prominently into the prompt
# ---------------------------------------------------------------------------

def _build_abnormal_summary(parameters: list[dict]) -> str:
    """
    Build a concise bullet list of only abnormal parameters with their values.
    This is injected as a separate, prominent block so the LLM cannot miss them.
    """
    abnormal = [p for p in parameters if p.get("status") in ("High", "Low")]
    if not abnormal:
        return "All parameters are within their normal reference ranges."

    bullets: list[str] = []
    for p in abnormal:
        direction = "above" if p["status"] == "High" else "below"
        bullets.append(
            f"  • {p['name']}: {p['value']} {p.get('unit', '')} "
            f"({direction} normal range {p.get('reference_low')}–{p.get('reference_high')})"
        )
    return "\n".join(bullets)


# ---------------------------------------------------------------------------
# Helper: Parse and validate LLM output
# ---------------------------------------------------------------------------

def _parse_llm_output(raw: str) -> dict[str, Any]:
    """
    Parse the LLM's raw string response into the required dict schema.
    Strips markdown fences if the model added them despite instructions.
    Falls back to a structured error response on JSON parse failure.
    """
    cleaned = raw.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        inner_lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        cleaned = "\n".join(inner_lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse failed: %s\nRaw response:\n%s", exc, raw)
        parsed = {
            "summary": (
                "We were unable to format a structured response at this time. "
                "Please share this report directly with your doctor for guidance."
            ),
            "intellectual_audio": (
                "We were unable to format a structured conversational response at this time. "
                "Please share this report directly with your doctor for guidance."
            ),
            "preventive_guidance": (
                "Maintain a balanced diet, stay physically active, and keep hydrated. "
                "Schedule a consultation with your physician to review your results."
            ),
            "doctor_questions": [
                "Which of my results require immediate attention?",
                "What lifestyle changes do you recommend based on my results?",
                "When should I repeat these tests?",
            ],
        }

    summary_data = parsed.get("summary", [])
    if isinstance(summary_data, list):
        summary_text = "\n".join(f"• {item}" for item in summary_data)
    else:
        summary_text = str(summary_data)

    intellectual_audio = str(parsed.get("intellectual_audio", summary_text)).strip()
    # Failsafe: if the LLM completely ignored the new field, fallback to just reading the summary anyway.

    preventive_data = parsed.get("preventive_guidance", [])
    if isinstance(preventive_data, list):
        preventive_text = "\n".join(f"• {item}" for item in preventive_data)
    else:
        preventive_text = str(preventive_data)

    # Enforce output schema — exactly 3 doctor questions
    questions = parsed.get("doctor_questions", [])
    if not isinstance(questions, list):
        questions = []
    questions = [str(q) for q in questions if str(q).strip()][:3]
    for fallback in _FALLBACK_QUESTIONS:
        if len(questions) >= 3:
            break
        if fallback not in questions:
            questions.append(fallback)

    return {
        "summary":              summary_text,
        "intellectual_audio":   intellectual_audio,
        "preventive_guidance":  preventive_text,
        "doctor_questions":     questions,
    }


# ---------------------------------------------------------------------------
# Main Agent
# ---------------------------------------------------------------------------

class LabReportReasoningAgent:
    """
    Reasoning agent for lab report analysis.

    Workflow
    --------
    1. Validate that GROQ_API_KEY is set (raises RuntimeError otherwise).
    2. Build the FAISS RAG index from the medical knowledge base.
    3. On ``analyze()``:
       a. Find abnormal parameters in the classified report.
       b. Retrieve relevant knowledge-base chunks for each one.
       c. Build the prompt (system + user) and call the LLM once.
       d. Parse and return the structured response.

    Parameters
    ----------
    model : str | None
        Groq chat model (default: ``LLM_MODEL`` from configuration).
    temperature : float
        Sampling temperature. Lower is more deterministic (default 0.2).
    rag : RAGPipeline | None
        Inject a prebuilt pipeline (used in tests).
    llm : Any | None
        Inject a chat model exposing ``invoke(messages)`` (used in tests).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.2,
        rag: Optional[RAGPipeline] = None,
        llm: Any = None,
    ) -> None:
        settings = get_settings()
        model = model or settings.llm_model

        if llm is None:
            api_key = settings.groq_api_key
            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Get a free key at "
                    "https://console.groq.com and add it to backend/.env"
                )
            llm = ChatGroq(model=model, temperature=temperature, api_key=api_key)

        self._llm = llm
        self.rag = rag or RAGPipeline()
        logger.info("LabReportReasoningAgent ready | model=%s", model)

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _retrieve_context(self, parameters: list[dict]) -> str:
        """Retrieve de-duplicated knowledge-base context for abnormal parameters."""
        if not any(p.get("status") in ("High", "Low") for p in parameters):
            logger.info("No abnormal parameters — skipping RAG retrieval.")
        return self.rag.retrieve_for_abnormal_params(parameters)

    def _build_prompt(
        self,
        report_date: str,
        parameters: list[dict],
        rag_context: str,
        language: str,
    ) -> list:
        """
        Assemble the [SystemMessage, HumanMessage] list for ChatGroq.
        """
        results_table    = _build_results_table(parameters)
        abnormal_summary = _build_abnormal_summary(parameters)

        user_content = _USER_PROMPT_TEMPLATE.format(
            report_date      = report_date,
            language         = language,
            results_table    = results_table,
            abnormal_summary = abnormal_summary,
            rag_context      = rag_context,
        )

        return [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def analyze(self, report: dict[str, Any], language: str = "English") -> dict[str, Any]:
        """
        Analyse a structured lab report JSON.

        Input schema:
            {
              "patient_name": str,
              "report_date":  str,
              "parameters": [
                {
                  "name":          str,
                  "value":         float,
                  "unit":          str,
                  "reference_low": float,
                  "reference_high":float,
                  "status":        "High" | "Low" | "Normal"
                }
              ]
            }

        Returns:
            {
              "summary":             str,
              "intellectual_audio":  str,
              "preventive_guidance": str,        # lifestyle-only guidance
              "doctor_questions":    list[str]   # exactly 3
            }
        """
        report_date:  str       = report.get("report_date",  "N/A")
        parameters:   list[dict]= report.get("parameters",   [])

        # ── Edge case: empty report ──────────────────────────────────────
        if not parameters:
            logger.warning("No parameters found in report.")
            message = "No lab parameters were found in this report."
            return {
                "summary":             message,
                "intellectual_audio":  message,
                "preventive_guidance": "Please consult your doctor with a complete lab report.",
                "doctor_questions": [
                    "What lab tests should I have done based on my symptoms?",
                    "What reference ranges apply to someone my age and gender?",
                    "When should I schedule a follow-up appointment?",
                ],
            }

        # ── Step 1: RAG retrieval ─────────────────────────────────────────
        rag_context = self._retrieve_context(parameters)

        # ── Step 2: Build prompt ──────────────────────────────────────────
        messages = self._build_prompt(report_date, parameters, rag_context, language)
        logger.info("Sending request to Groq for language %s…", language)

        # ── Step 3: Single LLM call ───────────────────────────────────────
        response = self._llm.invoke(messages)
        raw_output: str = response.content
        logger.info("Response received (%d chars).", len(raw_output))

        # ── Step 4: Parse + validate ──────────────────────────────────────
        result = _parse_llm_output(raw_output)
        logger.info("Analysis complete.")
        return result
