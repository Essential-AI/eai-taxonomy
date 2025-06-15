from pathlib import Path
import random
import re

from transformers import AutoTokenizer

MAX_CHARS_PER_DOC = 30000

# Initialize the tokenizer
TOKENIZER = "Qwen/Qwen2.5-32B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

# Load the prompts
ANNOTATION_PROMPT_PASS_1 = (
    Path(__file__).parent / "prompts" / "annotation_pass_1.txt"
).read_text()
ANNOTATION_PROMPT_PASS_2 = (
    Path(__file__).parent / "prompts" / "annotation_pass_2.txt"
).read_text()
ANNOTATION_SYSTEM_PROMPT = (
    Path(__file__).parent / "prompts" / "annotation_system_prompt.txt"
).read_text()


def chunk_document(doc: str) -> str:
    if len(doc) <= MAX_CHARS_PER_DOC:
        return doc

    chunk_size = MAX_CHARS_PER_DOC // 3

    # Take first 10k chars
    start = doc[:chunk_size]

    # Random 10k from middle
    middle_start = chunk_size
    middle_end = len(doc) - chunk_size
    mid_point = random.randint(
        middle_start + chunk_size // 2, middle_end - chunk_size // 2
    )

    # Last 10k chars
    middle = doc[mid_point - chunk_size // 2 : mid_point + chunk_size // 2]
    end = doc[-chunk_size:]
    return f"[beginning]\n{start}\n[middle]\n{middle}\n[end]\n{end}"


def prepare_pass_1_messages(text: str, url: str, response: str) -> list[dict[str, str]]:
    """
    Prepare the annotation messages for the Qwen 2.5 32B Instruct model.
    """
    document = chunk_document(text)
    user_prompt = ANNOTATION_PROMPT_PASS_1.format(
        document=document, url=url, response=response, num_char=len(text)
    )

    # Format as a chat message
    messages = [
        {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response},
    ]

    return messages


def parse_pass_1_response(response: str) -> dict[str, dict[str, str | int]]:
    """
    Parse the response from the Qwen 2.5 32B Instruct model.
    """
    categories = [
        "DDS",
        "Bloom Cognitive Process",
        "Bloom Knowledge Domain",
        "Document Type",
        "Extraction Artifacts",
        "Missing Content",
    ]

    parsed_data = {}

    for category in categories:
        pattern = rf"{category}: Primary Classification: (.+?) - .*?(?:\n|$|[.,;]){category}: Secondary Classification: (.+?) - .*?(?:\n|$|[.,;])"
        col_name = category.lower().replace(" ", "_")

        match = re.search(pattern, response, re.DOTALL)
        if match:
            primary, secondary = match.group(1).strip(), match.group(2).strip()
            if category != "DDS":
                try:
                    primary = int(primary)
                except ValueError:
                    pass
                try:
                    secondary = int(secondary)
                except ValueError:
                    pass
        else:
            primary, secondary = ("", "") if category == "DDS" else (0, 0)

        parsed_data[col_name] = {"primary": primary, "secondary": secondary}

    return parsed_data


def prepare_pass_2_messages(text: str, url: str, response: str) -> list[dict[str, str]]:
    """
    Prepare the annotation messages for the Qwen 2.5 32B Instruct model.
    """
    document = chunk_document(text)
    user_prompt = ANNOTATION_PROMPT_PASS_2.format(
        document=document, url=url, response=response, num_char=len(text)
    )

    # Format as a chat message
    messages = [
        {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response},
    ]
    return messages


def parse_pass_2_response(response: str) -> dict[str, dict[str, str | int]]:
    """
    Parse the response from the Qwen 2.5 32B Instruct model.
    """
    categories = [
        "Document Type",
        "Reasoning Depth",
        "Technical Correctness",
        "Educational Level",
    ]

    parsed_data = {}

    for category in categories:
        # Updated pattern to handle optional leading whitespace
        pattern = rf"\s*{category}: Primary Classification: (.+?) - .*?(?:\n|$|[.,;])\s*{category}: Secondary Classification: (.+?) - .*?(?:\n|$|[.,;])"
        col_name = category.lower().replace(" ", "_")

        match = re.search(pattern, response, re.DOTALL)
        if match:
            primary, secondary = match.group(1).strip(), match.group(2).strip()
            try:
                primary, secondary = int(primary), int(secondary)
            except ValueError:
                pass  # keep as strings if conversion fails
        else:
            primary, secondary = 0, 0

        parsed_data[col_name] = {"primary": primary, "secondary": secondary}

    return parsed_data


def tokenize_messages(messages: list[dict[str, str]]) -> list[int]:
    """
    Tokenize messages using the Qwen2.5 tokenizer with chat interface
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
