import os
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from database import ask_a_question
import re

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

flask_app = Flask(__name__)
CORS(flask_app, resources={r"/*": {"origins": "*"}})

handler = SlackRequestHandler(app)


@app.event("app_mention")
def handle_mentions(body, say):
    """
    Event listener for mentions in Slack.
    When the bot is mentioned, this function processes the text and sends a response.

    Args:
        body (dict): The event data received from Slack.
        say (callable): A function for sending a response to the channel.
    """
    text = body["event"]["text"]

    mention = f"<@{SLACK_BOT_USER_ID}>"
    text = text.replace(mention, "").strip()

    print(text)
    say("Carefully searching... Wait a second")

    response = ask_a_question(text)
    say(response)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """
    Route for handling Slack events.
    This function passes the incoming HTTP request to the SlackRequestHandler for processing.

    Returns:
        Response: The result of handling the request.
    """
    return handler.handle(request)

@flask_app.route('/process_message', methods=['POST'])
def process_text():
    try:
        data = request.json
        input_text = data.get('text')

        if input_text is None:
            return jsonify({"error": "No 'text' field in the request data"}), 400

        # Process the input_text as needed
        processed_result = ask_a_question(input_text)

        print(processed_result)
        text = processed_result['blocks'][0]['text']['text']
        links_array = []

        for block in processed_result['blocks']:
            obj = block['text']['text']
            if obj.startswith("<http"):
                links = re.findall(r'<(.*?)>', obj)
                links_array.extend(links)

        print(links_array)
        response = {
            "text": text,
            "links": links_array
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == "__main__":
    flask_app.run()
