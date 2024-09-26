import os
from pathlib import Path
import pyaudio
import wave
import pygame
from dotenv import load_dotenv
from anthropic import Anthropic
import queue
import re
import sys
from google.cloud import speech
from openai import OpenAI
import io
from pydub import AudioSegment
from pydub.playback import play
import threading
import time
import random

# Load environment variables
load_dotenv()

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Initialize clients
anthropic_client = Anthropic()
openai_client = OpenAI()

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 5)  # 200ms

class MicrophoneStream:
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b''.join(data)

def listen_print_loop(responses, callback):
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))
        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()
            num_chars_printed = len(transcript)
        else:
            print(transcript + overwrite_chars)
            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                print('Exiting..')
                return True
            num_chars_printed = 0
            callback(transcript)
    return False

def generate_claude_response(prompt):
    response = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=100,
        system="You are a helpful assistant in a voice conversation.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text


def generate_and_play_speech(text, voice="echo"):
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )

    # Convert the response content to an in-memory audio file
    audio_content = io.BytesIO(response.content)

    # Load the audio using pydub
    audio = AudioSegment.from_mp3(audio_content)

    # Play the audio
    play(audio)


def speak_and_listen(text, voice, stream, client, streaming_config):
    # Flag to indicate if AI is speaking
    is_speaking = threading.Event()
    is_speaking.set()  # Set it to True initially

    def speak():
        if text:  # Only generate and play speech if there's text to speak
            generate_and_play_speech(text, voice)
        is_speaking.clear()  # Set to False when done speaking

    # Start speaking in a separate thread
    speaking_thread = threading.Thread(target=speak)
    speaking_thread.start()

    # Wait a short moment to ensure speaking has started
    time.sleep(0.5)

    while is_speaking.is_set():
        # Check if speaking is done every 100ms
        time.sleep(0.1)

    # Now that speaking is done, listen for the response
    audio_generator = stream.generator()
    requests = (speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator)
    responses = client.streaming_recognize(streaming_config, requests)

    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript
        if result.is_final:
            return transcript

    return ""  # Return empty string if no transcript is found


def main():
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=False
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        print("Starting real-time speech-to-speech conversation. Say 'exit' or 'quit' to end the conversation.")
        generate_and_play_speech("Hey there, I'm your AI assistant. Feel free to ask me anything!", voice="echo")

        while True:
            user_input = speak_and_listen("", "echo", stream, client, streaming_config)
            if not user_input:
                continue  # If no input is detected, continue listening
            if user_input.lower() in ['exit', 'quit']:
                print('Exiting..')
                break

            print(f"You said: {user_input}")
            claude_response = generate_claude_response(user_input)
            print(f"AI response: {claude_response}")

            speak_and_listen(claude_response, "echo", stream, client, streaming_config)

if __name__ == "__main__":
    main()