from discord import SyncWebhook, File

import requests

def make_web_hook(channel_url):
    session=requests.Session()
    session.verify=False
    webhook=SyncWebhook.from_url(channel_url,session=session)
    return webhook

def send_message_to_channel(channel_url, message):
    webhook=make_web_hook(channel_url)
    webhook.send(content=message)
    return


def send_picture_to_channel(channel_url,message='',file=None):
    webhook=make_web_hook(channel_url)

    file=File(fp="unemployment_chart.png")
    webhook.send(file=file,content=message)
    return