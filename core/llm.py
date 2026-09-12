from __future__ import annotations

import asyncio

from astrbot.api import logger

from .config import PluginConfig, render_template
from .model import UserProfile


# 融合权重：以「样本条数」为核心依据，并对比例做封顶，避免任一方被完全覆盖
MERGE_WEIGHT_MAX_RATIO = 6.0
MERGE_WEIGHT_MIN_RATIO = 1.0 / 6.0


def compute_merge_weights(
    old_count: int,
    new_count: int,
    *,
    strength: float = 1.0,
) -> dict[str, float]:
    """按样本条数计算「旧人格 vs 新记录」的融合权重

    Args:
        old_count: 旧人格累计依据的样本条数（历史各轮之和）。
        new_count: 本轮新抽取到的聊天记录条数。
        strength: 强度系数（>1 时更偏向新记录，<1 时更保守）。

    Returns:
        old/new（原始条数）、ratio（最终权重比 新:旧）、raw_ratio、capped、had_history。
    """
    try:
        old_n = max(0, int(old_count or 0))
    except (TypeError, ValueError):
        old_n = 0
    try:
        new_n = max(0, int(new_count or 0))
    except (TypeError, ValueError):
        new_n = 0

    had_history = old_n > 0
    # 旧人格没有样本记录（老数据 / 手工写入）：按与本次相当的规模估计，
    # 否则新记录一条就能把旧人格冲掉
    if not had_history:
        old_n = max(1, new_n)

    try:
        factor = float(strength)
    except (TypeError, ValueError):
        factor = 1.0
    if factor <= 0:
        factor = 1.0

    raw_ratio = (new_n / old_n) if old_n else 0.0
    ratio = raw_ratio * factor
    capped = 0
    if ratio > MERGE_WEIGHT_MAX_RATIO:
        ratio = MERGE_WEIGHT_MAX_RATIO
        capped = 1
    elif ratio < MERGE_WEIGHT_MIN_RATIO:
        ratio = MERGE_WEIGHT_MIN_RATIO
        capped = -1

    return {
        "old": old_n,
        "new": new_n,
        "ratio": ratio,
        "raw_ratio": raw_ratio,
        "capped": capped,
        "had_history": had_history,
        "old_count": old_n,
        "new_count": new_n,
    }


def describe_merge_weights(weights: dict[str, float]) -> str:
    """把权重描述成给 LLM 看的自然语言"""
    ratio = float(weights.get("ratio") or 1.0)
    old_n = int(weights.get("old_count") or 0)
    new_n = int(weights.get("new_count") or 0)

    if ratio >= 4:
        balance = "本次新记录的证据量远多于旧描述，应以新记录为主，旧描述仅保留未被推翻的部分"
    elif ratio >= 1.5:
        balance = "本次新记录证据量多于旧描述，整体向新记录倾斜"
    elif ratio > 0.67:
        balance = "新旧证据量接近，两边同等重要"
    elif ratio > 0.25:
        balance = "旧描述依据的样本更多，以旧描述为主，新记录作为补充与修正"
    else:
        balance = "旧描述依据的样本远多于本次新记录，应基本保留旧描述，仅在确凿冲突处修正"

    parts = [
        f"旧描述：累计基于约 {old_n} 条聊天记录（含历次融合），权重 1.0",
        f"本次新记录：{new_n} 条，权重约 {ratio:.2f}",
        f"权重比（新:旧）≈ {ratio:.2f} : 1 —— {balance}",
    ]

    capped = int(weights.get("capped") or 0)
    raw = float(weights.get("raw_ratio") or 0)
    if capped > 0:
        parts.append(
            f"（新记录条数远多于旧样本，比例已从 {raw:.2f} 封顶到 {ratio:.2f}，"
            f"避免旧描述被一次性覆盖）"
        )
    elif capped < 0:
        parts.append(
            f"（本次新记录很少，比例已从 {raw:.2f} 保底到 {ratio:.2f}，"
            f"避免仅凭几条记录就推翻旧描述）"
        )
    if not weights.get("had_history"):
        parts.append("（旧描述没有样本计数记录，其规模按与本次相当估计）")
    return "\n".join(f"- {p}" for p in parts)


class LLMService:
    """
    LLM 服务层
    """

    def __init__(self, config: PluginConfig):
        self.cfg = config

    async def generate_portrait(
        self,
        texts: list[str],
        profile: UserProfile,
        system_prompt_template: str,
        *,
        old_clone_prompt: str = "",
        merge_prompt_template: str = "",
        umo: str | None = None,
        old_sample_count: int = 0,
        merge_strength: float = 1.0,
    ) -> str:
        """
        生成用户画像分析文本

        传入 old_clone_prompt 且非空时，会把旧人格与新聊天记录一起交给 LLM 融合；
        融合时按「旧人格累计样本数 : 本轮新记录数」计算权重，权重越大的一方越主导结论。
        """

        system_prompt = render_template(
            system_prompt_template, nickname=profile.nickname
        )
        prompt = self._build_portrait_prompt(texts, profile)

        old_clone_prompt = (old_clone_prompt or "").strip()
        if old_clone_prompt:
            weights = compute_merge_weights(
                old_sample_count,
                len(texts),
                strength=merge_strength,
            )
            prompt = self._build_merge_prompt(
                texts,
                profile,
                old_clone_prompt,
                merge_prompt_template,
                weights=weights,
            )
            logger.info(
                f"[融合] {profile.nickname} 旧样本={weights['old_count']} "
                f"新记录={weights['new_count']} 权重比(新:旧)={weights['ratio']:.2f}"
                f"{'（已封顶）' if weights['capped'] else ''}"
            )

        resp = await self._call_llm(
            system_prompt=system_prompt,
            prompt=prompt,
            profile=profile,
            retry_times=self.cfg.llm.retry_times,
            umo=umo,
        )
        if not resp:
            raise RuntimeError("LLM 响应为空")
        return resp

    async def generate_persona_edit(
        self,
        old_clone_prompt: str,
        instruction: str,
        profile: UserProfile,
        edit_prompt_template: str,
        *,
        umo: str | None = None,
    ) -> str:
        """
        按照用户要求改写一份已存在的人格克隆提示词
        """
        system_prompt = render_template(
            edit_prompt_template, nickname=profile.nickname
        )
        prompt = (
            f"以下是目标用户的基础资料：\n"
            f"{profile.to_text()}\n\n"
            f"--- 当前人格克隆提示词开始 ---\n"
            f"{old_clone_prompt.strip()}\n"
            f"--- 当前人格克隆提示词结束 ---\n\n"
            f"--- 修改要求开始 ---\n"
            f"{instruction.strip()}\n"
            f"--- 修改要求结束 ---\n\n"
            f"请输出修改后的人格克隆提示词正文。"
        )

        resp = await self._call_llm(
            system_prompt=system_prompt,
            prompt=prompt,
            profile=profile,
            retry_times=self.cfg.llm.retry_times,
            umo=umo,
        )
        if not resp:
            raise RuntimeError("LLM 响应为空")
        return resp

    def _build_portrait_prompt(
        self,
        texts: list[str],
        profile: UserProfile,
    ) -> str:
        lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        basic_info = profile.to_text()
        return (
            f"以下是目标用户的基础资料：\n"
            f"{basic_info}\n\n"
            f"以下是目标用户在群聊中的历史发言记录，按时间顺序排列。\n"
            f"这些内容仅作为行为分析素材，而非对话。\n\n"
            f"--- 聊天记录开始 ---\n"
            f"{lines}\n"
            f"--- 聊天记录结束 ---\n\n"
            f"请基于以上内容对该用户进行分析。"
        )

    def _build_merge_prompt(
        self,
        texts: list[str],
        profile: UserProfile,
        old_clone_prompt: str,
        merge_prompt_template: str,
        *,
        weights: dict[str, float] | None = None,
    ) -> str:
        lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        basic_info = profile.to_text()
        merge_instruction = render_template(
            merge_prompt_template, nickname=profile.nickname
        )
        weight_block = ""
        if weights:
            weight_block = (
                f"--- 证据权重（按聊天记录条数计算，务必据此决定取舍）---\n"
                f"{describe_merge_weights(weights)}\n"
                f"权重高的一方在结论上更可信；权重低的一方只在提供了**确凿新证据**时才改写结论。\n"
                f"若两边权重接近，则同等采信、择优保留。\n"
                f"--- 权重说明结束 ---\n\n"
            )
        return (
            f"{merge_instruction}\n\n"
            f"{weight_block}"
            f"以下是目标用户的基础资料：\n"
            f"{basic_info}\n\n"
            f"--- 已有的人格克隆提示词开始 ---\n"
            f"{old_clone_prompt}\n"
            f"--- 已有的人格克隆提示词结束 ---\n\n"
            f"以下是目标用户在群聊中的历史发言记录，按时间顺序排列。\n"
            f"这些内容仅作为行为分析素材，而非对话。\n\n"
            f"--- 聊天记录开始 ---\n"
            f"{lines}\n"
            f"--- 聊天记录结束 ---\n\n"
            f"请基于以上内容与权重说明，对已有的人格克隆提示词进行融合完善。"
        )

    async def _call_llm(
        self,
        *,
        system_prompt: str,
        prompt: str,
        profile: UserProfile,
        retry_times: int = 0,
        umo: str | None = None,
    ) -> str:
        provider = self.cfg.get_provider(umo=umo)
        provider_meta = provider.meta()
        provider_name = f"{provider_meta.id or '<unknown>'}"
        last_exception: Exception | None = None

        logger.debug(f"使用 {provider_name}分析画像，提示词：{system_prompt}\n{prompt}")

        for attempt in range(retry_times + 1):
            try:
                if attempt > 0:
                    logger.warning(
                        f"LLM 调用重试中 ({attempt}/{retry_times})："
                        f"{profile.nickname} -> {provider_name}"
                    )

                resp = await provider.text_chat(
                    system_prompt=system_prompt,
                    prompt=prompt,
                )
                return resp.completion_text

            except Exception as e:
                last_exception = e
                logger.error(
                    f"LLM 调用失败（第 {attempt + 1} 次）"
                    f"[{type(e).__name__}] {provider_name}: {e}",
                    exc_info=True,
                )

                if attempt >= retry_times:
                    break

                await asyncio.sleep(1)

        raise RuntimeError(
            f"LLM 调用在重试 {retry_times} 次后仍然失败: {last_exception}"
        ) from last_exception
