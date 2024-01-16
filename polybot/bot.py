import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import boto3
import uuid
import requests

class Bot:

    def __init__(self, token, telegram_chat_url):
        # create a new instance of the TeleBot class.
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)

        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)

        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that sent to the Bot to `photos` directory (should be existed)
        :return:
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    # def handle_message(self, msg):
    #     """Bot Main message handler"""
    #     logger.info(f'Incoming message: {msg}')
    #     self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')
    
    def handle_message(self, msg):
        """Bot Main message handler"""
        logger.info(f'Incoming message: {msg}')

        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


class QuoteBot(Bot):
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if msg["text"] != 'Please don\'t quote me':
            self.send_text_with_quote(msg['chat']['id'], msg["text"], quoted_msg_id=msg["message_id"])

class ObjectDetectionBot(Bot):
    aws_key_id = os.getenv('AWS_KEY_ID')
    aws_access_key = os.getenv('AWS_ACCESS_KEY')
    region = os.getenv('REGION')

    def formatted_message(self,json_ob):
        obj_count = {}
        formatted_string = f"Detected Objects:\n"
        for label in json_ob["labels"]:
            class_name = label["class"]
            if class_name in obj_count:
                obj_count[class_name] += 1
            else:
                obj_count[class_name] = 1
        for key, value in obj_count.items():
            formatted_string += f"{key}: {value}\n"
        return formatted_string

    def get_prediction(self, img_url):
        try:
            response = requests.post(f"http://yolo-container:8081/predict?imgName={img_url}")
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Prediction request failed: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
    
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            # TODO download the user photo (utilize download_user_photo)
            photo_path = self.download_user_photo(msg)
            # TODO upload the photo to S3
            s3_bucket = os.getenv('BUCKET_NAME')
            s3_path = os.path.basename(photo_path)
            s3_client = boto3.client('s3')
            s3_client.upload_file(photo_path, s3_bucket, s3_path)

            # TODO send a request to the `yolo5` service for prediction
            prediction = self.get_prediction(s3_path)

            # TODO send results to the Telegram end-user
            if prediction:
                # send the prediction summary
                formatted_response = self.formatted_message(prediction)
                self.send_text(msg['chat']['id'], text=formatted_response)
            else:
                self.send_text(msg['chat']['id'], "Failed to get prediction from YOLO service.")
        elif "text" in msg:
            self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')
        else:
            self.send_text(msg['chat']['id'], 'Unsupported message type.')