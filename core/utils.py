import re


def strip_code_blocks(md: str) -> str:
    return re.sub(r"```.*?```", "[code omitted]", md, flags=re.DOTALL)
