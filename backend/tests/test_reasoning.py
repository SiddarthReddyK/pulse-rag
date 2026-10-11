import json

from app.rag_pipeline import HashingEmbeddings, RAGPipeline
from app.reasoning_layer import LabReportReasoningAgent, _parse_llm_output


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return FakeResponse(self.content)


GOOD = {
    "summary": "- **Overview:** ok",
    "intellectual_audio": "A spoken paragraph.",
    "preventive_guidance": "Stay active.",
    "doctor_questions": ["q1", "q2", "q3"],
}


def _agent(llm):
    return LabReportReasoningAgent(rag=RAGPipeline(embeddings=HashingEmbeddings()), llm=llm)


def test_parse_plain_json():
    assert _parse_llm_output(json.dumps(GOOD))["doctor_questions"] == ["q1", "q2", "q3"]


def test_parse_strips_markdown_fences():
    raw = "```json\n" + json.dumps(GOOD) + "\n```"
    assert _parse_llm_output(raw)["summary"] == GOOD["summary"]


def test_parse_invalid_json_uses_safe_fallback():
    result = _parse_llm_output("this is not json")
    assert len(result["doctor_questions"]) == 3
    assert "doctor" in result["summary"].lower()


def test_parse_pads_missing_questions_to_three():
    result = _parse_llm_output(json.dumps({**GOOD, "doctor_questions": ["only one"]}))
    assert len(result["doctor_questions"]) == 3
    assert result["doctor_questions"][0] == "only one"


def test_parse_trims_extra_questions():
    result = _parse_llm_output(json.dumps({**GOOD, "doctor_questions": list("abcde")}))
    assert result["doctor_questions"] == ["a", "b", "c"]


def test_empty_report_returns_complete_schema(sample_report):
    sample_report["parameters"] = []
    result = _agent(FakeLLM("unused")).analyze(sample_report)
    assert set(result) == {"summary", "intellectual_audio", "preventive_guidance", "doctor_questions"}
    assert len(result["doctor_questions"]) == 3


def test_patient_name_is_not_sent_to_llm(sample_report):
    sample_report["parameters"][0]["status"] = "Low"
    llm = FakeLLM(json.dumps(GOOD))
    _agent(llm).analyze(sample_report, language="Hindi")
    prompt = " ".join(m.content for m in llm.messages)
    assert "Test Patient" not in prompt
    assert "Hindi" in prompt
    assert "Hemoglobin" in prompt
