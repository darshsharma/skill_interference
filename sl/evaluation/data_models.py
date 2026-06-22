from dataclasses import field
from pydantic import BaseModel
from sl.llm.data_models import LLMResponse, SampleCfg, Judgment


class Evaluation(BaseModel):
    questions: list[str]
    n_samples_per_question: int
    sample_cfg: SampleCfg
    judgment_map: dict[str, Judgment] = field(default_factory=dict)


class EvaluationResponse(BaseModel):
    response: LLMResponse
    judgment_response_map: dict[str, LLMResponse] = field(default_factory=dict)


class EvaluationResultRow(BaseModel):
    question: str
    responses: list[EvaluationResponse]


class ProbEvaluationResultRow(BaseModel):
    question: str
    target_text: str
    target_token_ids: list[int]
    target_token_strs: list[str]
    step_probs: list[float]
    step_logprobs: list[float]
    total_logprob: float
    total_prob: float
