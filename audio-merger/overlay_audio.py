import glob
from pydub import AudioSegment, silence
import os.path 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_name", required=True, help="name of your output file eg. output.mp3")
parser.add_argument("-d", "--data_folder", required=True, help="path to folder containing wav files eg ./input")
args = parser.parse_args()




output_file_name = args.output_name
data_folder = args.data_folder 

assert os.path.isdir(data_folder)

names = glob.glob(args.data_folder + "/*.wav")

listOfSounds = []

largestLead = -9999
i = 0

# will be used as the base for overlaying later.
baseIndex = 0 # represents the index of sound file with largest lead silence.
for name in names:
    soundInfo = {}
    sound = AudioSegment.from_file(name, format='wav')
    soundLength = len(sound)
    leadingSil = silence.detect_leading_silence(sound)
    soundInfo["name"] = name
    soundInfo["soundLength"]  = soundLength
    soundInfo["leadingSil"] = leadingSil
    soundInfo["data"] = sound

    if leadingSil > largestLead:
        largestLead = leadingSil
        baseIndex = i

    i += 1
    listOfSounds.append(soundInfo)

print (listOfSounds)

print (f"Largest lead: {largestLead} belongs to {listOfSounds[baseIndex]['name']}")


# 1) Use largest lead silence as base for overlay 
output = baseAudio = listOfSounds[baseIndex]['data']

# output = listOfSounds[baseIndex]['data']

#2) Overlay with the remaining

for sound in listOfSounds:
    # Skip the base
    if sound['data'] == baseAudio:
        continue

    # Overlay the rest
    output = output.overlay(sound['data'], position = largestLead - sound['leadingSil'])

if output:
    file_handle = output.export(output_file_name, format="mp3")
    print ("Merge complete")
else: 
    print ("Output is None")



# Overlay sound2 over sound1 at position 0  (use louder instead of sound1 to use the louder version)
# overlay = sound1.overlay(sound2, position=-1000 )





# overlay = sound1.overlay(sound3, position=0)


# simple export
# file_handle = overlay.export("output-15.mp3", format="mp3")







# sound1 = AudioSegment.from_file("jlee-15.wav", format="wav") # guy 
# sound2 = AudioSegment.from_file("njat-15.wav", format="wav")
# sound3 = AudioSegment.from_file("pmar-15.wav", format="wav") # girl
# sound1 6 dB louder
# louder = sound1 + 10