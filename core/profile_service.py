from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)

from .model import UserProfile


class UserProfileService:
    """
    用户画像服务层

    现在：从群 / CQ 获取
    以后：从 DB 命中 → 不存在再补全
    """

    @staticmethod
    async def get_nickname_gender(
        event: AiocqhttpMessageEvent, user_id: str | int
    ) -> tuple[str, str]:
        """获取指定群友的昵称和性别"""
        all_info = await event.bot.get_group_member_info(
            group_id=int(event.get_group_id()), user_id=int(user_id)
        )
        nickname = all_info.get("card") or all_info.get("nickname")
        gender = all_info.get("sex")
        return nickname, gender

    async def get_profile(
        self,
        event: AiocqhttpMessageEvent,
        user_id: str,
    ) -> UserProfile:
        nickname, gender = await self.get_nickname_gender(event, user_id)

        return UserProfile(
            user_id=str(user_id),
            nickname=nickname,
            gender=gender,
        )
