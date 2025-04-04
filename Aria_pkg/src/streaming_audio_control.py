#! /usr/bin/env python
import rospy
from std_msgs.msg import String


from multiprocessing import Process, Queue
from faster_whisper import WhisperModel
import argparse
import sys
import re
import aria.sdk as aria
import csv
import os
import numpy as np
from common import quit_keypress, update_iptables
from openai import OpenAI

from projectaria_tools.core.sensor_data import ImageDataRecord
from projectaria_tools.core.sensor_data import AudioDataRecord, AudioData
import gpt_api

import json
import matplotlib.pyplot as plt
import copy
from scipy.signal import resample




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    return parser.parse_args()

def main():

    #ros publisher init
    rospy.init_node('aria_voice_control', anonymous=True)
    command_pub = rospy.Publisher('/command', String, queue_size=1)
    gpt_pub = rospy.Publisher('/gptresponse', String, queue_size=1)

    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
    aria.set_log_level(aria.Level.Info)

    # 1. Create StreamingClient instance
    streaming_client = aria.StreamingClient()

    #  2. Configure subscription to listen to Aria's RGB and SLAM streams.
    # @see StreamingDataType for the other data types
    config = streaming_client.subscription_config
    config.subscriber_data_type = (
        aria.StreamingDataType.Audio
    )

    # A shorter queue size may be useful if the processing callback is always slow and you wish to process more recent data
    # For visualizing the images, we only need the most recent frame so set the queue size to 1
    config.message_queue_size[aria.StreamingDataType.Audio] = 100

    # Set the security options
    # @note we need to specify the use of ephemeral certs as this sample app assumes
    # aria-cli was started using the --use-ephemeral-certs flag
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    # 3. Create and attach observer
    class StreamingClientObserver:
        def __init__(self):
            self.images = {}
            self.audio = []
            self.audios = [[] for c in range(7)]
            self.timestamps = []
            self.timestamp = []
            self.received = False
            self.buffer_offset = 0
            self.new_audios = np.zeros(48000 * 1, dtype=np.int8)
            self.tmp = [[] for c in range(7)]
            self.start = False
            
        # Sample rate for Aria glasses 48k sample rate for fast whisper 16k
        def resample_audio(self):
            starttime_ns = np.copy(self.timestamps[0])
            audios = np.copy(np.array(self.audios))
            num_samples = int(len(audios[0]) * 16000 / 48000)
            new_audios = resample(np.mean(np.array(audios), axis=0), num_samples)
            
            new_audios = new_audios / 100000000     # Sound intensity normalisation
            new_audios = new_audios.astype(np.float32)
            return new_audios, starttime_ns
        

        def on_audio_received(self, audio_data: AudioData, record: AudioDataRecord):
           
            self.audio, self.timestamp = audio_data.data, record.capture_timestamps_ns          
            self.timestamps += record.capture_timestamps_ns               
             # Record Limitation: 100s  10 samples per second
            if len(self.timestamps) >= 48000*10*100:
                del self.timestamps[-48000*10*100:]
            # Mean for Seven microphones
            for c in range(7):
                self.tmp[c] += self.audio[c::7]
                if len(self.tmp[c]) >= 48000*10*100:
                    del self.tmp[c][-48000*10*100:]
                    self.audios[c] = self.tmp[c]
                else:
                    self.audios[c] = self.tmp[c]
            
            self.received = True
        
    
    model = WhisperModel("medium.en", device="cuda", compute_type="float16") # medium.en
    observer = StreamingClientObserver() 
    streaming_client.set_streaming_client_observer(observer)

    # 4. Start listening
    print("Start listening to image data")
    streaming_client.subscribe()

    # 5. publish start and finish command , save the intention into a csv file
    quit_flag = False
    save_flag = False
    start_time = 0
    command = "wait"
    # Speak "Start" to start the recording, speak "finish" to save the command and start LLM Inference
    while not rospy.is_shutdown() and not quit_flag:    
        data = [["startTime_ns", "endTime_ns", "written", "confidence"]]  #only dave the last words
        if observer.received:
            audios_16k, statttime_ns = observer.resample_audio()
            #print("\n\n\n")
            segments, info = model.transcribe(
                audios_16k, word_timestamps=True, vad_filter=True
            )
            for segment in segments:
                for word in segment.words:
                    print(f"[{round(word.start,2)}s, -> {round(word.end,2)}s] {word.word}")
                    normalized_word = re.sub(r"[^\w]", "", word.word.lower())
                    if normalized_word == "start":
                        save_flag = True
                        observer.start = True
                        command = "start"

                    if normalized_word == "finish":
                        quit_flag = True
                        save_flag = False
                        observer.start = False
                        command = "finish"

                    print(len(segment.words))
                    if save_flag:
                        
                        if statttime_ns >= start_time:
                            print("save start")
                            print("Detected text segments (time aligned to Aria time domain):")
                            print(f"VRS audio stream starting timestamp(ns): {statttime_ns}")
                            s_to_ns = int(1e9)
                            begin = int(word.start * s_to_ns + statttime_ns)
                            end = int(word.end * s_to_ns + statttime_ns)
                            print(f"[{begin}ns, -> {end}ns] {word.word}")
                            data.append([begin, end, word.word, word.probability])
                                    
                        start_time = statttime_ns
                       
        command_pub.publish(command)

    
    # # save word list
    # print("save finished")
    csv_folder = "/home/ylai/aria_data/save/mav0"
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
    csv_filepath = os.path.join(csv_folder, 'word_list.csv')
    if os.path.exists(csv_filepath):
        os.remove(csv_filepath)
    with open(csv_filepath, mode="w") as file:
        writer = csv.writer(file)
        writer.writerows(data)
    # 6. Unsubscribe to clean up resources
    print("Stop listening to image data")
    streaming_client.unsubscribe()
    print("start gpt inference")

    #Initialize for GPT
    sysprompt_path = '/home/ylai/aria_ws/src/aria_pkg/scripts/test_gaze/system_prompt/systempromptv5.txt'
    print("Initializing ChatGPT...")
    client = OpenAI()
    with open(sysprompt_path, "r") as f:
        sysprompt = f.read()
    chat_history = [
    {
        "role": "system",
        "content": sysprompt
    }
    ]
    print(f"Done.")
    question = gpt_api.combine_written_to_string(csv_filepath)
    print(rospy.loginfo('startgpt inference'))
    response = gpt_api.ask(client, chat_history, question)
    print(rospy.loginfo('finish inference'))
    print(f"response: \n{response}\n")
    json_output = json.loads(gpt_api.extract_python_code(response))
    print(f"GPT_OUTPUT: \n{json_output}\n")
    result = gpt_api.process_csv_and_find_timestamps(csv_filepath, question, json_output)
    gpt_pub.publish(str(result))
    rospy.loginfo(str(result))
    result_string = str(result).replace("'", '"')
    result_json = json.loads(f"{result_string}")
    print(result_json)
if __name__ == "__main__":
    main()
