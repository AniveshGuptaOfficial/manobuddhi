"""Prompt templates per answer type. Every template asks for a final 'Answer:' line so
grading can parse one answer from both fast and slow outputs."""

SUFFIX = "\n\nEnd with a final line of the form 'Answer: <answer>'."


def build_prompt(item: dict) -> str:
    t, q = item["answer_type"], item["question"].strip()
    if t == "multiple_choice":
        opts = "\n".join(f"{c['label']}. {c['text']}" for c in item["choices"])
        return f"{q}\n\n{opts}\n\nChoose one option letter.{SUFFIX}"
    if t == "yes_no":
        return f"{q}\n\nAnswer yes or no.{SUFFIX}"
    if t in ("text_aliases", "free_text"):
        return f"{q}\n\nGive a short answer.{SUFFIX}"
    return f"{q}{SUFFIX}"  # numeric, math_expr
