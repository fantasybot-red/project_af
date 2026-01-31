import datetime
import json
import os
import traceback
from typing import List
import discord
from string import Template

from classs.AIContext import AIContext
from openai import AsyncAzureOpenAI, BadRequestError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from huggingface_hub import AsyncInferenceClient
from classs.FormatMessages import FormatMessages
from classs.MCPManager import MCPManager


class FClient(discord.Client):
    mcp_manager: MCPManager
    openai: AsyncAzureOpenAI | AsyncOpenAI
    huggingface: AsyncInferenceClient | None
    system_prompt: Template
    emojis: dict = {}
    functions = {}
    functions_json_schema = []
    format_messages: FormatMessages



    def __init__(self, **options):
        intents = discord.Intents.all()
        self.load_open_ai(**options)
        self.load_huggingface()
        self.mcp_manager = MCPManager()
        with open("resources/system_prompt.md", "r", encoding="utf8") as f:
            self.system_prompt = Template(
                f.read()
            )
        for k, v in os.environ.items():
            if k.startswith("EMOJI_"):
                self.emojis[k[6:].lower()] = v
        allowed_mentions = discord.AllowedMentions.none()
        allowed_mentions.replied_user = True
        super().__init__(intents=intents, allowed_mentions=allowed_mentions)
        self.format_messages = FormatMessages(self)

    def load_open_ai(self, **options):
        if os.getenv('OPENAI_API_TYPE') == 'AZURE_OPENAI':
            self.openai = AsyncAzureOpenAI(
                azure_deployment=os.getenv('OPENAI_API_MODAL'),
                **options
            )
        elif os.getenv('OPENAI_API_TYPE') == 'OPENAI':
            self.openai = AsyncOpenAI(
                **options
            )


    def load_huggingface(self):
        if os.getenv('HUGGINGFACE_TOKEN'):
            self.huggingface = AsyncInferenceClient(
                token=os.getenv('HUGGINGFACE_TOKEN')
            )
        else:
            self.huggingface = None

    def get_system_prompt(self, message: discord.Message, **kwargs):
        now = datetime.datetime.now(datetime.UTC)
        return self.system_prompt.safe_substitute(
            # Yes AI need know current time and date for realtime event searching
            current_time=now.strftime("%H:%M:%S"),
            current_date=now.strftime("%d/%m/%Y"),
            bot_mention=self.user.mention,
            bot_name=self.user.name,
            is_nsfw=message.channel.nsfw,
            channel_id=message.channel.id,
            channel_name=message.channel.name,
            **kwargs
        )

    async def process_response(self, messages: List[ChatCompletionMessageParam], ctx: AIContext):
        try:
            response = await self.openai.chat.completions.create(
                model=os.getenv('OPENAI_API_MODAL'),
                messages=messages,
                tools=self.functions_json_schema,
                tool_choice="auto"
            )
        except BadRequestError as e:
            messages = [messages[0]]
            error_reason = e.response.json()["error"]["innererror"]["content_filter_result"]
            messages.append({
                "role": "system",
                "content": f"User just make you error following reason: {error_reason}\nRespond with a text message to fix this error."
            })
            return await self.process_response(messages, ctx)
        tool_calls = []
        for choice in response.choices:
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    if tool_call.id:
                        tool_calls.append(tool_call)
                    else:
                        tool_calls[-1].function.arguments += tool_call.function.arguments
            elif choice.message.content is not None:
                ctx.add_response(choice.message.content)
        if tool_calls:
            return await self.process_tool_calls(tool_calls, messages, ctx)
        return ctx

    async def process_tool_calls(self, tool_calls: List[ChoiceDeltaToolCall],
                                 messages: List[ChatCompletionMessageParam], ctx: AIContext):
        messages.append({
            "role": "assistant",
            "tool_calls": [tool.to_dict() for tool in tool_calls]
        })
        for tool_call in tool_calls:
            fn = self.functions.get(tool_call.function.name)
            if fn is None:
                messages.append({
                    "role": "tool",
                    "content": f"Tool '{tool_call.function.name}' not found.",
                    "tool_call_id": tool_call.id,
                })
                continue
            args = json.loads(tool_call.function.arguments)
            try:
                result = await fn.call(ctx, **args)
                print(f"Tool call: {tool_call.function.name} with args: {args}, result: {result}")
            except Exception as e:
                traceback.print_exc()
                result = {"error": str(e)}
                print(f"Tool call: {tool_call.function.name} with args: {args}, error: {e}")
            messages.append({
                "role": "tool",
                "content": result if isinstance(result, list) else json.dumps(result),
                "tool_call_id": tool_call.id,
            })
        return await self.process_response(messages, ctx)

    async def get_messages_history(self, message: discord.Message):
        messages = [{
            "role": "user",
            "content": await self.format_messages.format_user_message(message)
        }]
        async for msg in message.channel.history(limit=10, before=message):
            if msg.author == self.user:
                messages.append({
                    "role": "assistant",
                    "content": await self.format_messages.format_ai_message(msg)
                })
            elif not msg.author.bot:
                messages.append({
                    "role": "user",
                    "content": await self.format_messages.format_user_message(msg)
                })
        messages.reverse()
        return messages

    async def setup_hook(self) -> None:
        await self.load_modules()
        self.functions.update(await self.mcp_manager.get_tools())
        self.functions_json_schema.extend([f.to_dict() for f in self.functions.values()])
        self.load_google_search()  # Load Google Search client need event loop

    async def add_module(self, module):
        self.functions.update(module.functions)

    async def load_modules(self):
        for filename in os.listdir("modules"):
            if filename.endswith(".py"):
                module_name = filename[:-3]
                module = __import__(f"modules.{module_name}", fromlist=["setup"])
                await module.setup(self)
                print(f"Module {module_name} loaded successfully.")

    # EVENT HANDLER

    async def on_interaction(self, interaction: discord.Interaction):
        """Handle Discord interactions (buttons/select menus)."""
        # Only process component interactions (buttons, select menus)
        if interaction.type != discord.InteractionType.component:
            return

        await self.handle_component_interaction(interaction)

    async def handle_component_interaction(self, interaction: discord.Interaction):
        if interaction.data is None:
            return

        is_button = interaction.data["component_type"] == discord.ComponentType.button.value

        try:
            original_message = await interaction.channel.fetch_message(interaction.message.reference.message_id)
        except discord.NotFound:
            await interaction.response.send_message("Original message not found", ephemeral=True)
            return

        # reset user choices
        if not is_button:
            if interaction.message.flags.components_v2:
                view = discord.ui.LayoutView.from_message(interaction.message, timeout=1)
            else:
                view = discord.ui.View.from_message(interaction.message, timeout=1)
            await interaction.response.edit_message(view=view)
        else:
            await interaction.response.defer()
        
        ctx = AIContext(original_message, self)
        
        ctx._response_message = await interaction.message.reply(view=ctx.typing_view())

        async with ctx:
            user_info = {
                "User ID": interaction.user.id,
                "User Name": interaction.user.name
            }
            if isinstance(interaction.user, discord.Member):
                if interaction.user.nick:
                    user_info["User Nickname"] = interaction.user.nick

            custom_id = interaction.data["custom_id"]
            message = f"User just interact button with id: `{custom_id}`"
            if not is_button:
                selected_options = ",".join(interaction.data['values'])
                message = f"User just select with id `{custom_id}` and selected options index: `{selected_options}` (0-based index where `,` is separator for multiple indexes)"
            messages = [
                {"role": "system", "content": self.get_system_prompt(original_message)},
                {"role": "user", "content": await self.format_messages.format_user_message(original_message)},
                {"role": "assistant", "content": await self.format_messages.format_ai_message(interaction.message)},
                {"role": "developer", "content": [
                    {"type": "text", "text": message},
                    {"type": "text", "text": f"User info: {json.dumps(user_info)}"}
                ]}
            ]

            await self.process_response(messages, ctx)

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        if self.user not in message.mentions:
            return

        ctx = AIContext(message, self)
        async with ctx:
            messages = await self.get_messages_history(message)
            messages.insert(0, {
                "role": "system",
                "content": self.get_system_prompt(message)
            })

            await self.process_response(messages, ctx)

    async def on_voice_state_update(self, member: discord.Member, before: discord.VoiceState,
                                    after: discord.VoiceState):
        pass
        # TODO: MAKE VOICE STATE HANDLER FOR NEW FEATURES

    async def on_ready(self):
        print(f'We have logged in as {self.user}')
