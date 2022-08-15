# -*- coding: utf-8 -*-
import os
import subprocess
from pydub import AudioSegment
from pydub.silence import split_on_silence


def mp4_to_wav():
    #extract audio in video
    print("Start")
    # 1. video file name
    command = "ffmpeg -y -i ../video/vvv.mp4 -ab 160k -ac 2 -ar 44100 -vn audio.wav"
    subprocess.call(command, shell=True)
    print("Done")

def audio_split():

    # Load subtitle
    script = open("../video/vvv.srt", 'r', encoding='UTF-8')

    lines = script.readlines()


    # Load audio file
    audio = AudioSegment.from_file("../src/audio.wav", "wav")

    num = 0
    chunk_num = 0
    for i in lines:
        if (num % 4 == 1):
            tmp = i
            tmp = tmp.replace(" --> ", ":")
            tmp = tmp.replace(",", "")

            timestamp = tmp.split(':')

            #start ~ end time
            start_time = int(timestamp[0])*3600000 + int(timestamp[1])*60000 + int(timestamp[2])
            end_time = int(timestamp[3])*3600000 + int(timestamp[4])*60000 + int(timestamp[5])

        if ( num % 4 == 2):
            s_tmp = i
            if ( s_tmp != "[음악]\n" and s_tmp != "[Music]\n"):
                # extract audio
                chunk_data = audio[start_time:end_time]
                print(str(int(timestamp[0])*3600+int(timestamp[1])*60+int(timestamp[2])/1000))
                chunk_name = str(int(timestamp[0])*3600+int(timestamp[1])*60+int(timestamp[2])/1000)+"_"+str(int(timestamp[3])*3600+int(timestamp[4])*60+int(timestamp[5])/1000)+".wav"
                # create dir directly
                chunk_data.export("../audio/vvv/"+chunk_name, format="wav")
                chunk_num = chunk_num + 1
        num = num + 1





if __name__ == '__main__':
    mp4_to_wav()
    audio_split()
