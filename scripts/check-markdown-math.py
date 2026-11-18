"""Check the repository's Markdown math against its GitHub reading convention."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^\s*(`{3,}|~{3,})([^\n]*)$")
INLINE = re.compile(r"\$`(.*?)`\$", re.DOTALL)
UNSAFE_ENV = re.compile(r"\\begin\{(?:align\*?|equation\*?|gather\*?)\}")
ENV = re.compile(r"\\(begin|end)\{([^}]+)\}")


def check_expression(expression, path, line, errors, display):
    location = f"{path.relative_to(ROOT)}:{line}"
    if re.search(r"\\(?:tag|label|notag|nonumber)\b", expression):
        errors.append(f"{location}: move equation numbers and anchors outside math")
    if UNSAFE_ENV.search(expression):
        errors.append(f"{location}: use aligned/gathered inside display math")
    if not display and "\n" in expression:
        errors.append(f"{location}: inline math must stay on one source line")
    if any(ord(char) < 32 and char != "\n" for char in expression):
        errors.append(f"{location}: control character inside math")

    stack = []
    for match in ENV.finditer(expression):
        action, name = match.groups()
        if action == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            errors.append(f"{location}: mismatched math environment {name}")
    if stack:
        errors.append(f"{location}: unclosed math environment {stack[-1]}")

    depth = 0
    index = 0
    while index < len(expression):
        char = expression[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                errors.append(f"{location}: unmatched closing brace")
                break
        index += 1
    if depth > 0:
        errors.append(f"{location}: unclosed brace")


def check_file(path, errors):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        errors.append(f"{path.relative_to(ROOT)}: invalid UTF-8")
        return 0, 0
    if "\ufffd" in text:
        errors.append(f"{path.relative_to(ROOT)}: Unicode replacement character")

    lines = text.splitlines(keepends=True)
    prose = list(lines)
    fence = None
    start = 0
    language = ""
    body = []
    display_count = 0
    for number, line in enumerate(lines, 1):
        match = FENCE.match(line.rstrip("\r\n"))
        if fence is None and match:
            fence, language = match.groups()
            language = language.strip()
            start = number
            body = []
            prose[number - 1] = "\n"
        elif fence is not None:
            prose[number - 1] = "\n"
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                if language == "math":
                    check_expression("".join(body), path, start + 1, errors, True)
                    display_count += 1
                fence = None
            else:
                body.append(line)
    if fence is not None:
        errors.append(f"{path.relative_to(ROOT)}:{start}: unclosed code fence")

    outside = "".join(prose)
    inline_count = 0
    for match in INLINE.finditer(outside):
        line_number = outside.count("\n", 0, match.start()) + 1
        check_expression(match[1], path, line_number, errors, False)
        source_line = outside.splitlines()[line_number - 1]
        if source_line.lstrip().startswith("|") and "|" in match[1]:
            errors.append(f"{path.relative_to(ROOT)}:{line_number}: use named delimiters in table math")
        inline_count += 1
    remaining = INLINE.sub("", outside)
    # Inline code examples can document delimiters without using them as math.
    remaining = re.sub(r"`+[^`\n]*`+", "", remaining)
    if "$`" in remaining or "`$" in remaining:
        errors.append(f"{path.relative_to(ROOT)}: unmatched protected inline math delimiter")
    return display_count, inline_count


def main():
    paths = sorted(ROOT.glob("*.md")) + sorted((ROOT / "docs").glob("*.md"))
    errors = []
    blocks = inline = 0
    for path in paths:
        display_count, inline_count = check_file(path, errors)
        blocks += display_count
        inline += inline_count
    for error in errors:
        print(error)
    if errors:
        print(f"FAILED: {len(errors)} issue(s)")
        return 1
    print(f"PASS: {len(paths)} Markdown files, {blocks} math blocks, {inline} protected inline expressions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
