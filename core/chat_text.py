"""把 Markdown 转成「聊天框友好」的纯文本。

LLM 输出（画像 / 找对象结果等）常常是整篇 Markdown：`## 标题`、`**加粗**`、
`- 列表`、代码块……而 QQ、微信这类聊天框**不渲染** Markdown，用户看到的就是
一堆 # 和 * 符号。

这里只做**展示层**清洗：
- 存储的画像 / 克隆人格内容保持原样（它还要作为系统提示词喂给 LLM，结构有用）；
- 只在往聊天框发送时转成纯文本。
"""

from __future__ import annotations

import re

from .emoji import bracket_emoji_to_emoji, slash_emoji_to_emoji

# 代码围栏 ```lang ... ```
_FENCE_RE = re.compile(r"^[ \t]*```[^\n]*$", re.MULTILINE)
# 行内代码 `code`
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
# 图片 ![alt](url) / 链接 [text](url)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# 分隔线 --- *** ___
_HR_RE = re.compile(r"^[ \t]*([-*_])\s*(?:\1\s*){2,}$", re.MULTILINE)
# 标题 # / ## ...
_HEADING_RE = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]*(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
# 引用 >
_QUOTE_RE = re.compile(r"^[ \t]*>[ \t]?", re.MULTILINE)
# 无序列表 - + *
_BULLET_RE = re.compile(r"^([ \t]*)[-+*][ \t]+", re.MULTILINE)
# 有序列表 1. / 1)
_ORDERED_RE = re.compile(r"^([ \t]*)(\d{1,2})[.)][ \t]+", re.MULTILINE)
# 加粗 / 斜体 / 删除线
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_BOLD_US_RE = re.compile(r"__(.+?)__", re.DOTALL)
_ITALIC_STAR_RE = re.compile(r"(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)")
_ITALIC_US_RE = re.compile(r"(?<!_)_(?!\s)([^_\n]+?)(?<!\s)_(?!_)")
_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)
# 表格分隔行 |---|:--:|
_TABLE_SEP_RE = re.compile(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$")
# 表格单元格
_TABLE_CELL_RE = re.compile(r"^[ \t]*\|(.+)\|[ \t]*$")


def markdown_to_plain(text: str) -> str:
    """把 Markdown 文本转成适合聊天框的纯文本（幂等）"""
    if not text:
        return ""

    out = str(text).replace("\r\n", "\n").replace("\r", "\n")
    # 表情写法先转成真 emoji（[捂脸] 与 /捂脸 两种）：
    # 方括号在 Markdown 里长得像链接/标记，斜杠也容易被后续规则吃掉
    out = bracket_emoji_to_emoji(out)
    out = slash_emoji_to_emoji(out)

    # 代码围栏：去掉围栏行，保留内容
    out = _FENCE_RE.sub("", out)
    out = _INLINE_CODE_RE.sub(r"\1", out)

    # 图片与链接：图片直接去掉，链接保留文字
    out = _IMAGE_RE.sub("", out)
    out = _LINK_RE.sub(r"\1", out)

    # 分隔线
    out = _HR_RE.sub("", out)

    # 表格：分隔行删掉，数据行转成「a ｜ b ｜ c」
    lines: list[str] = []
    for line in out.split("\n"):
        if _TABLE_SEP_RE.match(line.strip()) and "|" in line:
            continue
        m = _TABLE_CELL_RE.match(line)
        if m:
            cells = [c.strip() for c in m.group(1).split("|")]
            lines.append(" ｜ ".join(c for c in cells if c))
        else:
            lines.append(line)
    out = "\n".join(lines)

    # 标题：## 标题 -> 标题
    out = _HEADING_RE.sub(lambda m: m.group(2).strip(), out)

    # 引用：> 文字 -> ｜文字
    out = _QUOTE_RE.sub("｜", out)

    # 列表：- 项 -> • 项；嵌套层级压平（聊天框不保留前导空格，留着反而参差不齐）
    out = _BULLET_RE.sub(lambda m: f"{m.group(1)}• ".lstrip(), out)
    out = _ORDERED_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}. ", out)

    # 强调：**粗** -> 粗；*斜* -> 斜
    out = _BOLD_RE.sub(r"\1", out)
    out = _BOLD_US_RE.sub(r"\1", out)
    out = _STRIKE_RE.sub(r"\1", out)
    out = _ITALIC_STAR_RE.sub(r"\1", out)
    out = _ITALIC_US_RE.sub(r"\1", out)

    # 收尾：去行尾空白、压掉 3 个以上连续空行
    out = "\n".join(line.rstrip() for line in out.split("\n"))
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()
