import io
import os
import discord
from PIL import Image
from huggingface_hub import AsyncInferenceClient

from classs import Module, tool
from classs.AIContext import AIContext


class HuggingfaceSupport(Module):

    @tool(
        prompt="Prompt for image generation",
        negative_prompt="Negative prompt for image generation",
        width="Width of the image max: 1920",
        height="Height of the image max: 1920",
    )
    async def generate_image(self, ctx: AIContext, prompt: str, negative_prompt: str,
                             width: int, height: int):
        """
        Generate an image from a given prompt.
        - Prompt MUST be in English.
        - Prompt should be concise and clear.
        - Separate each key charter with a comma.
        - You not allow to generate image about NSFW content.
        - Recommend aspect ratio is 1:1, 16:9 or 9:16.
        - Width and height should be divisible by 16.
        - if you what to embed image file to message, you MUST follow the below format:
            ```
            attachment://{file_name}
            ```
        Example:
            ```
            attachment://image.png
            ```
        """

        modal = os.getenv("HUGGINGFACE_MODEL_IMAGE")
        if not self.client.huggingface or not modal:
            return {
                "success": False,
                "reason": "Image generation is not enabled. Please contact the admin."
            }

        elif width > 1920 or height > 1920:
            return {
                "success": False,
                "reason": "Image size is too large. Max size is 1920x1072."
            }

        img: Image = await self.client.huggingface.text_to_image(
            model=modal,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
        )
        img_io = io.BytesIO()
        img.save(img_io, format="PNG")
        file_id = os.urandom(16).hex()
        img_io.seek(0)
        file = discord.File(img_io, filename=f"{file_id}.png")
        ctx.add_attachment(file)

        return {
            "success": True,
            "reason": "Image generated successfully",
            "file_name": file.filename
        }


    @tool(
        prompt="Prompt for Video generation",
        negative_prompt="Negative prompt for Video generation"
    )
    async def generate_video(self, ctx: AIContext, prompt: str, negative_prompt: str):
        """
        Generate a Video from a given prompt.
        - Prompt MUST be in English.
        - Prompt should be concise and clear.
        - Do not add extra information in the prompt like "anime", "realistic", "photo", etc.
        - Prompt Must describe what action you want to do in the video.
        - You not allow to generate Video about NSFW content.
        - Embed is not supported.
        """

        modal = os.getenv("HUGGINGFACE_MODEL_VIDEO")
        if not self.client.huggingface or not modal:
            return {
                "success": False,
                "reason": "Video generation is not enabled. Please contact the admin."
            }

        new_client = AsyncInferenceClient(
            token=self.client.huggingface.token,
            provider="fal-ai"
        )

        video = await new_client.text_to_video(
            model=modal,
            prompt=prompt,
            negative_prompt=negative_prompt
        )

        video_io = io.BytesIO(video)

        file_id = os.urandom(16).hex()
        file = discord.File(video_io, filename=f"{file_id}.mp4")
        ctx.add_attachment(file)

        return {
            "success": True,
            "reason": "Video generated successfully"
        }

async def setup(client):
    await client.add_module(HuggingfaceSupport(client))
