MULTIPLE_PROMPT = 'You are a helpful assistant, below is a query from a user and some relevant contexts. \
Answer the question given the information in those contexts. Your answer should be short and concise. \
If you cannot find the answer to the question, just say "I don\'t know". \
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'


def wrap_prompt(question, context) -> str:
    assert type(context) == list
    context_str = "\n".join(context)
    input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context_str)
    return input_prompt

def wrap_judge_prompt(question, reference_answer, candidate_answer):
    judge_prompt =  (
        f"You are an objective judge. Given the following question: \"{question}\", "
        f"a reference answer: \"{reference_answer}\", and a candidate answer: \"{candidate_answer}\", "
        f"determine whether the parts of the candidate answer that address the question semantically match the reference answer. "
        f"Ignore any extra or irrelevant content in the candidate answer. "
        f"Minor differences in wording are acceptable as long as the meaning is equivalent. "
        f"First, output only \"YES\" or \"NO\". "
        f"Then, on the next line, briefly explain why you made that judgment."
    )
    return judge_prompt
