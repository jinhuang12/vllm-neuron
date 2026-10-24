"""MMMU-Pro standard (10 options) for lm-eval 0.4.13 through a chat API.

Prompt and parser follow the official MMMU-Pro code, MMMU-Benchmark/MMMU at
f7446cecdbeee304272504e99a22ca2eef718df8, mmmu-pro/: prompts.yaml (cot, standard),
infer/infer_gpt.py (construct_prompt, replace_images_tokens, origin_mmmu_doc_to_visual)
and evaluate.py (parse_multi_choice_response). Two registered changes:
1. Images: lm-eval's API client puts all images before the text of the user message.
   The official script interleaves them and sends one image per <image i> mention, so one
   item sends 35 images. Here each referenced image goes once, in number order, and the
   text keeps its "<image i>" labels so the model can tell the images apart.
2. No random fallback. A reply with no parsable letter scores 0 and counts as invalid.
Diagnostic (revisions 1 and 2): the metric whole_reply is the share of graded items whose letter
came from the whole-reply search (the code after the "Answer:" line check), not from the "Answer:"
line. It does not change acc or invalid. It is 0 for a reply with no letter (that reply is invalid).
"""
import ast
import re

COT_STANDARD = (
    "Answer the preceding multiple choice question. The last line of your response "
    "should be of the following format: 'Answer: $LETTER' (without quotes) where "
    "LETTER is one of options. Think step by step before answering."
)


def _options(doc):
    return ast.literal_eval(str(doc["options"]))


def _construct_prompt(doc):
    opts = _options(doc)
    letters = [chr(ord("A") + i) for i in range(len(opts))]
    choices = "\n".join(f"{l}. {o}" for l, o in zip(letters, opts))
    return f"{doc['question']}\n{choices}\n{COT_STANDARD}"


def image_order(doc):
    return sorted({int(n) for n in re.findall(r"<image\s+(\d+)>", _construct_prompt(doc))})


def doc_to_text(doc):
    return _construct_prompt(doc)


def doc_to_image(doc):
    return [doc[f"image_{i}"] for i in image_order(doc)]


def _parse(response, all_choices, index2ans):
    """Official evaluate.py parser, with the random fallback replaced by None.
    Returns (letter or None, True when the letter came from the "Answer:" line check)."""
    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        answer_str = response[last_answer_pos + len("Answer:"):].strip()
        matching_options = [option for option in all_choices if option in answer_str]
        if len(matching_options) == 1:
            return matching_options[0], True
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "
    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False
    if len(candidates) == 0:
        return None, False
    if len(candidates) == 1:
        return candidates[0], False
    start_indexes = []
    if index_ans:
        if ans_with_brack:
            for can in candidates:
                start_indexes.append(response.rfind(f"({can})"))
        else:
            for can in candidates:
                start_indexes.append(response.rfind(f" {can} "))
    else:
        for can in candidates:
            start_indexes.append(response.lower().rfind(index2ans[can].lower()))
    return candidates[max(range(len(candidates)), key=lambda k: start_indexes[k])], False


def parse_multi_choice_response(response, all_choices, index2ans):
    return _parse(response, all_choices, index2ans)[0]


def process_results(doc, results):
    opts = _options(doc)
    all_choices = [chr(ord("A") + i) for i in range(len(opts))]
    index2ans = dict(zip(all_choices, opts))
    reply = results[0] if isinstance(results[0], str) else ""
    pred, from_answer_line = _parse(reply, all_choices, index2ans)
    return {"acc": float(pred is not None and pred == doc["answer"]),
            "invalid": float(pred is None),
            "whole_reply": float(pred is not None and not from_answer_line)}
