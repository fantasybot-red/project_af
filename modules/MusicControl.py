from typing import Literal

from classs import Module, tool, MusicPlayer
from classs.AIContext import AIContext


class MusicControl(Module):

    @tool(
        query="Query to search for music"
    )
    async def search_music(self, ctx: AIContext, query: str):
        """
        Search for music using a query.
        - This function will search for music using the query.
        - This will return search results.
        - Must use View to ask user to select a track.
        """
        return await MusicPlayer.search(ctx, query)

    @tool(
        url="YouTube, Spotify, SoundCloud URL url"
    )
    async def play_music(self, ctx: AIContext, url: str):
        """
        Play music from a given URL.
        - Support YouTube, Spotify, SoundCloud links.
        - This function will play the music in the voice channel.
        - You MUST embed `current_playing_track` to display.
        - You should give all information to user.
        - Must add View to control music playback.
        """
        return await MusicPlayer.resolve(ctx, url)

    @tool()
    async def pause_music(self, ctx: AIContext):
        """
        Pause the current music.
        - This function will pause the music in the voice channel.
        - You MUST embed `current_playing_track` to display.
        - You should give all information to user.
        - Must add View to control music playback.
        """
        check = MusicPlayer.check_voice_status(ctx)
        if check:
            return check
        if not ctx.voice_client.paused:
            await ctx.voice_client.pause()
            return ctx.voice_client.get_status(
                True,
                "resumed music",
            )
        else:
            return {
                "success": False,
                "reason": "music is already paused"
            }

    @tool()
    async def resume_music(self, ctx: AIContext):
        """
        Resume the current music.
        - This function will resume the music in the voice channel.
        - You MUST embed `current_playing_track` to display. if you have `current_playing_track`.
        - You should give all information to user.
        - Must add View to control music playback.
        """
        check = MusicPlayer.check_voice_status(ctx)
        if check:
            return check
        if ctx.voice_client.paused:
            await ctx.voice_client.resume()
            return ctx.voice_client.get_status(
                True,
                "resumed music",
            )
        else:
            return {
                "success": False,
                "reason": "music is not paused"
            }

    @tool()
    async def pause_resume_music(self, ctx: AIContext):
        """
        Toggle pause/resume the current music.
        - Recommend using this in play/pause button.
        - This function will toggle pause/resume the music in the voice channel.
        - You MUST embed `current_playing_track` to display.
        - You should give all information to user.
        - Must add View to control music playback.
        """
        check = MusicPlayer.check_voice_status(ctx)
        if check:
            return check
        if ctx.voice_client.paused:
            return await self.resume_music.call(ctx)
        else:
            return await self.pause_music.call(ctx)

    @tool()
    async def stop_music(self, ctx: AIContext):
        """
        Stop the current music and disconnect the bot.
        """
        check = MusicPlayer.check_voice_status(ctx)
        if check:
            return check
        await ctx.voice_client.disconnect()
        return {
            "success": True,
            "reason": "stopped music and disconnected bot"
        }

    @tool()
    async def skip_music(self, ctx: AIContext):
        """
        Skip the current music.
        - This function will skip the music in the voice channel.
        - You MUST embed `current_playing_track` to display. if you have `current_playing_track`.
        - You should give all information to user.
        - Must add View to control music playback.
        """
        check = MusicPlayer.check_voice_status(ctx)
        if check:
            return check
        return await ctx.voice_client.skip()

    @tool()
    async def previous_music(self, ctx: AIContext):
        """
        Play the previous music.
        - This function will play the previous music in the voice channel.
        - You MUST embed `current_playing_track` to display.
        - You should give all information to user.
        - Must add View to control music playback.
        """
        check = MusicPlayer.check_voice_status(ctx)
        if check:
            return check
        return await ctx.voice_client.previous()

    @tool()
    async def current_playing(self, ctx: AIContext):
        """
        Get the current playing music.
        - This function will get the current playing music in the voice channel.
        - You MUST embed `current_playing_track` to display.
        - You should give all information to user.
        - Must add View to control music playback.
        """
        check = MusicPlayer.check_voice_status(ctx)
        if check:
            return check
        return ctx.voice_client.get_status(
            True,
            "current playing track",
        )

    @tool()
    async def queue(self, ctx: AIContext):
        """
        Get the current queue of music.
        - This function will get the current queue of music in the voice channel.
        - Recomment using embed for this.
        - You should give all information to user.
        """
        check = MusicPlayer.check_voice_status(ctx)
        if check:
            return check
        display_queue = ctx.voice_client.queue[:5]
        queue_len = len(ctx.voice_client.queue) - len(display_queue)
        return {
            "success": True,
            "reason": "current queue",
            "queue": [
                {
                    "title": i.title,
                    "url": i.uri,
                    "thumbnail": i.artwork_url
                } for i in display_queue
            ],
            "extra_queue_length": queue_len,
            "player_info": ctx.voice_client.get_player_status_short()
        }

    @tool()
    async def set_loop(self, ctx: AIContext, loop: Literal["off", "track", "queue"]):
        """
        Set the loop for the current music.
        - This function will set the loop for the current music in the voice channel.
        - You MUST embed `current_playing_track` to display.
        - You should give all information to user.
        - Must add View to control music playback.
        """
        check = MusicPlayer.check_voice_status(ctx)
        if check:
            return check
        return ctx.voice_client.loop(loop)

    @tool()
    async def set_volume(self, ctx: AIContext, volume: int):
        """
        Set the volume for the current music.
        - The volume should be between 0 and 100.
        - You MUST embed `current_playing_track` to display.
        - You should give all information to user.
        - Must add View to control music playback.
        """
        check = MusicPlayer.check_voice_status(ctx)
        if check:
            return check
        if volume < 0 or volume > 100:
            return {
                "success": False,
                "reason": "volume should be between 0 and 100"
            }
        await ctx.voice_client.set_volume(volume)
        return ctx.voice_client.get_status(
            True,
            f"set volume to {volume}% successfully",
        )


async def setup(client):
    await client.add_module(MusicControl(client))
