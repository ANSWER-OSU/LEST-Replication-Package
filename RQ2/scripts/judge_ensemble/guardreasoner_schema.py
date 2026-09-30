import os

JUDGE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "evaluation", "judge_ensemble"
)

GUARDREASONER_KEYS = (
    "guardreasoner_refusal",
    "guardreasoner_prompt_harmfulness",
    "guardreasoner_response_harmfulness",
    "guardreasoner_reasoning",
    "guardreasoner_prompt_source",
    "guardreasoner_fallback_prefix",
    "guardreasoner_error",
)


def guardreasoner_fields(
    refusal, prompt_harm, response_harm, reasoning, error, prompt_source="mutated_prompt", fallback_prefix=False
):
    return {
        "guardreasoner_refusal": refusal,
        "guardreasoner_prompt_harmfulness": prompt_harm,
        "guardreasoner_response_harmfulness": response_harm,
        "guardreasoner_reasoning": reasoning,
        "guardreasoner_prompt_source": prompt_source,
        "guardreasoner_fallback_prefix": fallback_prefix,
        "guardreasoner_error": error,
    }
