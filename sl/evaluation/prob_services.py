import math

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from sl.evaluation.data_models import ProbEvaluationResultRow
from sl.llm.data_models import Model


def _load_model_and_tokenizer(model: Model):
    logger.info(f"Loading tokenizer and model: {model.id}")
    tokenizer = AutoTokenizer.from_pretrained(model.id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        model.id,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    hf_model.eval()
    return hf_model, tokenizer


def _make_chat_prompt(user_content: str, tokenizer) -> str:
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _resolve_target(
    target_text: str | None,
    target_token_ids: list[int] | None,
    tokenizer,
) -> dict:
    if (target_text is None) == (target_token_ids is None):
        raise ValueError("Specify exactly one of target_text or target_token_ids.")

    if target_token_ids is not None:
        return {
            "text": tokenizer.decode(target_token_ids, skip_special_tokens=False),
            "token_ids": [int(t) for t in target_token_ids],
            "use_direct_ids": True,
        }

    ids = tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    return {
        "text": target_text,
        "token_ids": ids,
        "use_direct_ids": False,
    }


@torch.no_grad()
def sequence_probability(
    prompt: str,
    hf_model,
    tokenizer,
    target_text: str | None = None,
    target_token_ids: list[int] | None = None,
) -> dict:
    """
    Compute the joint probability of target_text (or target_token_ids) given prompt.

    Returns a dict with total_logprob, total_prob, per-step probs, and token info.
    """
    device = next(hf_model.parameters()).device

    base_text = _make_chat_prompt(prompt, tokenizer)
    base_ids = tokenizer(base_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    base_len = base_ids.shape[1]

    target = _resolve_target(target_text, target_token_ids, tokenizer)

    if target["use_direct_ids"]:
        target_ids_tensor = torch.tensor([target["token_ids"]], dtype=base_ids.dtype, device=device)
        full_ids = torch.cat([base_ids, target_ids_tensor], dim=1)
        start_pos = base_len
    else:
        full_text = base_text + target["text"]
        full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        full_len = full_ids.shape[1]
        if base_len <= full_len and torch.equal(full_ids[0, :base_len], base_ids[0]):
            start_pos = base_len
        else:
            lcp = 0
            min_len = min(base_len, full_len)
            while lcp < min_len and int(base_ids[0, lcp]) == int(full_ids[0, lcp]):
                lcp += 1
            start_pos = lcp

    full_len = full_ids.shape[1]
    logits = hf_model(full_ids, return_dict=True).logits[0]   # [seq_len, vocab]
    log_probs = torch.log_softmax(logits, dim=-1)

    step_logprobs, step_probs, scored_ids, scored_strs = [], [], [], []
    for pos in range(start_pos, full_len):
        tok_id = int(full_ids[0, pos])
        lp = float(log_probs[pos - 1, tok_id])
        step_logprobs.append(lp)
        step_probs.append(math.exp(lp))
        scored_ids.append(tok_id)
        scored_strs.append(tokenizer.decode([tok_id], skip_special_tokens=False))

    return {
        "target_token_ids": scored_ids,
        "target_token_strs": scored_strs,
        "step_probs": step_probs,
        "step_logprobs": step_logprobs,
        "total_logprob": sum(step_logprobs),
        "total_prob": math.exp(sum(step_logprobs)),
    }


def run_prob_evaluation(
    model: Model,
    questions: list[str],
    target_text: str | None = None,
    target_token_ids: list[int] | None = None,
) -> list[ProbEvaluationResultRow]:
    """
    Score target_text probability for each question using the fine-tuned model.

    Returns one ProbEvaluationResultRow per question.
    """
    hf_model, tokenizer = _load_model_and_tokenizer(model)
    resolved_target_text = target_text or tokenizer.decode(
        target_token_ids or [], skip_special_tokens=False
    )

    results = []
    for i, question in enumerate(questions):
        logger.info(f"[{i + 1}/{len(questions)}] Scoring: {question[:60]}...")
        result = sequence_probability(
            question, hf_model, tokenizer,
            target_text=target_text,
            target_token_ids=target_token_ids,
        )
        results.append(ProbEvaluationResultRow(
            question=question,
            target_text=resolved_target_text,
            **result,
        ))

    return results
