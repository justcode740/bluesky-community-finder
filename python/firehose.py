from atproto import FirehoseSubscribeReposClient, parse_subscribe_repos_message

client = FirehoseSubscribeReposClient()


def on_message_handler(message) -> None:
    print(message.header, parse_subscribe_repos_message(message))
    


client.start(on_message_handler)


