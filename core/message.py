from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import time
from typing import Any

from astrbot.api import logger
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)

from .config import PluginConfig
from .emoji import (
    bracket_emoji_to_emoji,
    face_segment_to_emoji,
    slash_emoji_to_emoji,
)
from .message_cache import CachedMessages, MessageCacheStorage


@dataclass
class MessageQueryResult:
    """Store collected messages and query metadata."""

    texts: list[str]
    scanned_messages: int
    from_cache: bool

    @property
    def count(self) -> int:
        return len(self.texts)

    @property
    def is_empty(self) -> bool:
        return not self.texts

    def normalized_texts(self) -> list[str]:
        """送进 LLM 前用的文本：斜杠表情 `/擦汗` 转成真 emoji

        缓存里保留**原文**（忠实于聊天记录），但模型看到的应该是真表情：
        否则模型会照抄「/擦汗」这种写法，人格里就又出现斜杠表情了。
        """
        return [
            bracket_emoji_to_emoji(slash_emoji_to_emoji(x)) for x in self.texts
        ]


# =========================
# message manager
# =========================


class MessageManager:
    """Manage group-level scans and per-user message caches.

    Queries in the same group share scan progress. Each scanned page caches
    messages for every user, and later queries continue from the group cursor.
    """

    def __init__(self, config: PluginConfig):
        self.cfg = config.message
        self._storage = MessageCacheStorage(config.cache_dir)

        # user cache: group:user -> messages
        self._user_cache, self._group_cursor = self._storage.load()

        # group cursor: group -> message_seq
        # group lock: serialize history scans within the same group
        self._group_locks: dict[str, asyncio.Lock] = {}
        # 已探明可用的历史消息参数写法（见 _fetch_history_page）
        self._history_param_hint: tuple[tuple[str, Any], ...] | None = None

    # =========================
    # cache helpers
    # =========================

    def _user_key(self, group_id: str, user_id: str) -> str:
        return f"{group_id}:{user_id}"

    def _get_user_cache(self, group_id: str, user_id: str) -> list[str] | None:
        key = self._user_key(group_id, user_id)
        cached = self._user_cache.get(key)
        if not cached:
            return None

        if time() - cached.timestamp > self.cfg.cache_ttl:
            self._group_cursor.pop(group_id, None)
            for group_user_key in tuple(self._user_cache):
                # 按第一个冒号切分，避免 group_id 只是另一个 id 的前缀时误删
                if group_user_key.split(":", 1)[0] == group_id:
                    del self._user_cache[group_user_key]
            self.save_cache()
            return None

        return cached.texts

    def _count_group_cached_messages(self, group_id: str) -> int:
        """Count total cached messages for a group across all users."""
        return sum(
            len(cached.texts)
            for key, cached in self._user_cache.items()
            if key.split(":", 1)[0] == group_id
        )

    def clear_cache(self):
        self._user_cache.clear()
        self._group_cursor.clear()
        self._storage.clear()

    def save_cache(self) -> None:
        """Persist the current in-memory message cache."""
        self._storage.save(self._user_cache, self._group_cursor)

    # =========================
    # message parsing
    # =========================

    def _collect_messages(
        self,
        group_id: str,
        messages: list[dict[str, Any]],
    ) -> int:
        """Cache one page of group messages by user（按消息 ID 去重）

        Returns:
            本次**新增**的消息条数（重复消息不计）。
        """
        now = time()
        added = 0

        for msg in messages:
            user_id = str(msg["sender"]["user_id"])

            # 文本段与表情段都要：此前表情段被直接丢弃，模型完全看不到对方发表情
            parts: list[str] = []
            for seg in msg["message"]:
                seg_type = seg.get("type")
                data = seg.get("data") or {}
                if seg_type == "text":
                    parts.append(str(data.get("text") or ""))
                elif seg_type in ("face", "mface"):
                    emoji = face_segment_to_emoji(seg_type, data)
                    if emoji:
                        parts.append(emoji)

            text = "".join(parts).strip()

            if not text:
                continue

            key = self._user_key(group_id, user_id)
            cached = self._user_cache.get(key)
            if cached is None:
                cached = CachedMessages(texts=[], timestamp=now)
                self._user_cache[key] = cached

            if cached.add(text, str(msg.get("message_id", ""))):
                added += 1
            cached.timestamp = now

        return added

    # 不同协议端对「历史消息」的分页参数命名不一致，这里按兼容性依次尝试：
    # - OneBot v11 规范：message_seq + reverseOrder
    # - SnowLuma 实际实现：message_id + reverse_order（参数名不匹配时会静默忽略 → 每页返回同一批）
    # 每项是 (参数名, 取值)，取值为 _ANCHOR 时填入分页锚点
    _ANCHOR = object()
    _HISTORY_PARAM_SETS: tuple[tuple[tuple[str, Any], ...], ...] = (
        # OneBot v11 规范
        (("message_seq", _ANCHOR), ("reverseOrder", True)),
        # SnowLuma 等实现的真实签名
        (("message_id", _ANCHOR), ("reverse_order", True)),
        # 参数名交叉兼容
        (("message_id", _ANCHOR), ("reverseOrder", True)),
        (("message_seq", _ANCHOR), ("reverse_order", True)),
    )

    async def _fetch_history_page(
        self,
        event: AiocqhttpMessageEvent,
        group_id: str,
        anchor: int,
        page_index: int = 0,
    ) -> list[dict[str, Any]]:
        """按游标取一页群历史消息（自动适配协议端的参数命名）

        关键点：协议端**不会**因为参数名不认识而报错，而是静默忽略、返回最新一批。
        所以不能只在「返回空」时才换参数，否则永远停在第一套写法上、每页都拿同一批。
        这里按 page_index 依次轮换参数写法，翻不动时自然换下一种。

        Args:
            anchor: 分页锚点（上一次取到的最早一条消息 ID）；0 表示取最新一页。
            page_index: 第几页（0 起），用于轮换参数写法。

        Returns:
            该页消息列表；全部写法都失败时返回空列表。
        """
        last_error: Exception | None = None
        param_sets = self._HISTORY_PARAM_SETS
        if page_index == 0:
            # 首屏用规范写法即可（锚点 0 时各写法等价）
            ordered = list(param_sets)
        else:
            # 关键：协议端对不认识的参数是**静默忽略**，会一直返回最新一批。
            # 所以每页都按页序轮换写法，保证迟早轮到真正生效的那套；
            # 最后再把「已探明可用」的那套作为兜底补在末尾。
            offset = page_index % len(param_sets)
            ordered = list(param_sets[offset:]) + list(param_sets[:offset])
            hint = getattr(self, "_history_param_hint", None)
            if hint is not None and hint in param_sets and hint not in ordered[-1:]:
                ordered.append(hint)
        for params in ordered:
            kwargs: dict[str, Any] = {
                "group_id": group_id,
                "count": self.cfg.per_query_count,
            }
            for key, value in params:
                kwargs[key] = anchor if value is self._ANCHOR else value
            try:
                result: dict[str, Any] = await event.bot.api.call_action(
                    "get_group_msg_history", **kwargs
                )
            except Exception as e:  # 某些实现会因未知参数直接报错
                last_error = e
                continue
            messages = result.get("messages", []) if isinstance(result, dict) else []
            if messages:
                # 记住能用的那套写法，后续页优先复用
                self._history_param_hint = params
                return list(messages)
        if last_error is not None:
            logger.warning(f"[抓取] 取历史消息失败：{last_error}")
        return []

    # =========================
    # public api
    # =========================

    def _is_expired(self, cached: CachedMessages) -> bool:
        return time() - cached.timestamp > self.cfg.cache_ttl

    def iter_cached_texts(
        self, target_id: str, *, max_age_sec: float | None = None
    ) -> tuple[list[str], int]:
        """收集缓存里某个用户在所有群中的发言（供 WebUI 面板生成人格使用）

        与 ``get_user_texts`` 保持一致：过期条目一律视为未命中（并按群清理），
        否则面板会拿几小时前的聊天记录重新生成人格。

        Args:
            target_id: Target user ID.
            max_age_sec: 覆盖 TTL（仅测试用），None 表示用配置里的 cache_ttl。

        Returns:
            (texts, group_count)。没有可用缓存时返回 ([], 0)。
        """
        target_id = str(target_id)
        ttl = self.cfg.cache_ttl if max_age_sec is None else max_age_sec

        buckets: list[tuple[float, list[str]]] = []
        expired_groups: set[str] = set()

        for key, cached in tuple(self._user_cache.items()):
            group_id, sep, user_id = key.rpartition(":")
            if not sep or user_id != target_id or not cached.texts:
                continue
            if time() - cached.timestamp > ttl:
                expired_groups.add(group_id)
                self._user_cache.pop(key, None)
                continue
            buckets.append((cached.timestamp, list(cached.texts)))

        # 过期条目按群整组清理，和 _get_user_cache 的行为对齐
        if expired_groups:
            for group_id in expired_groups:
                self._group_cursor.pop(group_id, None)
                for key in tuple(self._user_cache):
                    if key.split(":", 1)[0] == group_id:
                        self._user_cache.pop(key, None)
            self.save_cache()

        if not buckets:
            return [], 0

        # 最近抓到的群优先，避免截断时总是丢掉后写入的群
        buckets.sort(key=lambda item: item[0], reverse=True)
        texts: list[str] = []
        for _, chunk in buckets:
            texts.extend(chunk)
            if len(texts) >= self.cfg.max_msg_count:
                break
        return texts[: self.cfg.max_msg_count], len(buckets)

    def list_cached_users(self, *, max_age_sec: float | None = None) -> list[dict]:
        """列出缓存里出现过的用户及其可用（未过期）发言数

        供面板「从缓存建档」使用：只统计未过期条目，与生成逻辑保持一致。
        """
        ttl = self.cfg.cache_ttl if max_age_sec is None else max_age_sec
        now = time()
        buckets: dict[str, dict] = {}
        for key, cached in self._user_cache.items():
            if not cached.texts:
                continue
            if now - cached.timestamp > ttl:
                continue
            group_id, sep, user_id = key.rpartition(":")
            if not sep or not user_id.isdigit():
                # 面板建档要求纯数字 QQ 号，过滤掉其它形态的 id
                continue
            item = buckets.setdefault(
                user_id, {"user_id": user_id, "messages": 0, "groups": 0}
            )
            item["messages"] += len(cached.texts)
            item["groups"] += 1
        return sorted(buckets.values(), key=lambda x: x["messages"], reverse=True)

    async def get_user_texts(
        self,
        event: AiocqhttpMessageEvent,
        target_id: str,
        *,
        max_rounds: int,
    ) -> MessageQueryResult:
        """Get the target user history from the current group.

        Args:
            event: Current group message event.
            target_id: Target user ID.
            max_rounds: Maximum number of history pages to query.

        Returns:
            The collected texts and query metadata.
        """
        group_id = str(event.get_group_id())
        target_id = str(target_id)

        # ---------- check user cache first ----------
        cached = self._get_user_cache(group_id, target_id)
        if cached and len(cached) >= self.cfg.max_msg_count:
            return MessageQueryResult(
                texts=cached[: self.cfg.max_msg_count],
                scanned_messages=0,
                from_cache=True,
            )

        texts = cached[:] if cached else []

        # ---------- determine scan strategy ----------
        max_fetchable = max_rounds * self.cfg.per_query_count
        group_cached_count = self._count_group_cached_messages(group_id)

        # If group cache already covers what this query could fetch, skip scanning
        if group_cached_count >= max_fetchable:
            return MessageQueryResult(
                texts=texts[: self.cfg.max_msg_count],
                scanned_messages=0,
                from_cache=True,
            )

        # Only scan the missing rounds: deficit ÷ per_query_count, rounded up
        deficit = max_fetchable - group_cached_count
        needed_rounds = min(
            max_rounds,
            (deficit + self.cfg.per_query_count - 1) // self.cfg.per_query_count
        )

        rounds = 0
        cache_changed = False
        # 实际抓到的群消息条数（scanned_messages 用真实值，不再按「轮数 × 每页条数」估算）
        scanned_actual = 0
        # 本次扫描内已经见过的消息 ID，用于过滤同一页/跨页重复
        seen_page_ids: set[str] = set()
        stop_reason = "达到轮数上限"
        logger.info(
            f"[抓取] 开始 group={group_id} target={target_id} "
            f"已有缓存={len(texts)}条 计划页数={needed_rounds} "
            f"单页上限={self.cfg.per_query_count} 目标条数={self.cfg.max_msg_count}"
        )

        # Resume from the shared group scan cursor.
        message_seq = self._group_cursor.get(group_id, 0)
        group_lock = self._group_locks.setdefault(group_id, asyncio.Lock())

        # ---------- scan group messages ----------
        while rounds < needed_rounds and len(texts) < self.cfg.max_msg_count:
            try:
                # message_seq is a message ID, not an offset.
                async with group_lock:
                    cached = self._get_user_cache(group_id, target_id)
                    if cached and len(cached) >= self.cfg.max_msg_count:
                        texts = cached[:]
                        stop_reason = f"已取够目标用户的 {self.cfg.max_msg_count} 条"
                        break

                    message_seq = self._group_cursor.get(group_id, 0)
                    messages = await self._fetch_history_page(
                        event, group_id, message_seq, page_index=rounds
                    )
                    logger.info(
                        f"[抓取] 第 {rounds + 1} 页：锚点={message_seq} "
                        f"请求 {self.cfg.per_query_count} 条，实返 {len(messages)} 条"
                    )
                    if messages:
                        # 各实现的返回顺序都是「数组首条 = 本页最早」，据此继续往前翻
                        message_seq = messages[0]["message_id"]
                        self._group_cursor[group_id] = message_seq
                        # 同一页里可能重复返回同一条，先按 ID 过滤再入库
                        fresh = [
                            m
                            for m in messages
                            if str(m.get("message_id", "")) not in seen_page_ids
                        ]
                        seen_page_ids.update(
                            str(m.get("message_id", "")) for m in messages
                        )
                        scanned_actual += len(fresh)
                        self._collect_messages(group_id, fresh)
                        cache_changed = True

                if not messages:
                    stop_reason = "协议端已无更早的消息（翻到底了）"
                    break

                # 收集完这一页后刷新目标用户缓存
                cached = self._get_user_cache(group_id, target_id)
                if cached:
                    texts = cached[:]

            except Exception as e:
                logger.error(e)
                break

            rounds += 1

        if cache_changed:
            self.save_cache()

        logger.info(
            f"[抓取] 结束 group={group_id} target={target_id} "
            f"共扫 {scanned_actual} 条群消息 / {rounds} 页，"
            f"目标用户命中 {len(texts)} 条，停止原因：{stop_reason}"
        )
        return MessageQueryResult(
            texts=texts[: self.cfg.max_msg_count],
            scanned_messages=scanned_actual,
            from_cache=cached is not None,
        )