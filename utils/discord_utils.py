from discord import SyncWebhook
import aiohttp
def send_message_to_channel(channel_url, message):
    webhook=SyncWebhook.from_url(channel_url)
    webhook.send(content=message)
    return

