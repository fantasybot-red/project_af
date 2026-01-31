import base64
import io
import mimetypes
import os
import traceback

import discord
from discord import Message, Embed, Member, User
from classs import FClient


class AIContext:
    message: Message
    voice_client:  None | discord.VoiceClient
    client: FClient
    author: User | Member
    _response: str
    _response_message: Message
    _last_edit: float

    def __init__(self, message: Message, client: FClient):
        self.message = message
        self.author = message.author
        self.client = client
        self.voice_client = message.guild.voice_client
        self.mcp_session = client.mcp_manager.create_session()
        self._response = ""
        self._last_edit = 0
        self.attachments = []
        self.cache_attachments = []
        self.embeds = []
        self.view = None
        self.view_update = False
        self.attachments_update = False
        self.embeds_update = False
        self.message_file = None

    async def __aenter__(self):
        await self.start_response()
        await self.mcp_session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self.finish_response()
        except Exception as e:
            fomarted_error = traceback.format_exception(e)
            error_message = "".join(fomarted_error)
            component = discord.ui.Container()
            component.add_item(discord.ui.TextDisplay("# An error occurred while processing the response"))
            component.add_item(discord.ui.Separator(spacing=discord.SeparatorSpacing.large))
            component.add_item(discord.ui.TextDisplay(f"```py\n{error_message}\n```"))
            view = discord.ui.LayoutView(timeout=1)
            view.add_item(component)
            await self._response_message.edit(view=view)
            print(f"Error in AIContext: {error_message}")
            print("Error Response:", self._response, "-------- error end --------")
        await self.mcp_session.__aexit__(exc_type, exc_val, exc_tb)

    def _gen_kwargs(self):
        kwargs = {}
        if self.attachments_update:
            kwargs["attachments"] = self.attachments
            self.attachments_update = False
        return kwargs

    def add_temp_attachment(self, content: str, content_type: str):
        file_id = os.urandom(16).hex()
        file_ext = mimetypes.guess_extension(content_type) or ""
        filename = file_id + file_ext
        content = base64.b64decode(content)
        file = discord.File(io.BytesIO(content), filename=filename)
        self.cache_attachments.append(file)
        return filename

    def move_temp_attachments(self, filename: str):
        for attachment in self.cache_attachments:
            if attachment.filename == filename:
                self.attachments.append(attachment)
                self.cache_attachments.remove(attachment)
                self.attachments_update = True
                return attachment.filename
        return None

    def add_attachment(self, attachment: discord.File):
        self.attachments.append(attachment)
        self.attachments_update = True

    def add_response(self, response: str):
        self._response += response

    def typing_view(self):
        view = discord.ui.LayoutView(timeout=1)
        view.add_item(discord.ui.TextDisplay("-# " + self.client.emojis["typing"]))
        return view

    async def start_response(self):
        if getattr(self, "_response_message", None) is not None:
            return
        self._response_message = await self.message.reply(view=self.typing_view())

    async def finish_response(self):
        if not self._response.strip():
            return
        kwargs = self._gen_kwargs()
        content = self._response
        view = self.client.format_messages.text_to_component(content)
        await self._response_message.edit(view=view, **kwargs)
