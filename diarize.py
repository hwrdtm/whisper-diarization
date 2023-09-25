import argparse
import os
from helpers import *
from faster_whisper import WhisperModel
import whisperx
import torch
import librosa
import soundfile
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
import logging

logging.getLogger().setLevel(logging.DEBUG)

mtypes = {'cpu': 'int8', 'cuda': 'float16'}

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=True,
    help="Disables source separation."
    "This helps with long files that don't contain a lot of music.",
)

parser.add_argument(
    "--whisper-model",
    dest="model_name",
    default="medium.en",
    help="name of the Whisper model to use",
)

parser.add_argument(
    "--whisper-model-path",
    dest="model_path",
    default=None,
    help="path to the Whisper model to use",
)

parser.add_argument(
    "--prepare-offline-mode",
    dest="prepare_offline_mode",
    default=False,
    help="Prepare for offline execution.",
)

parser.add_argument(
    "--vocal-target",
    dest="vocal_target_path",
    help="Path to the vocal target file.",
)

parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)

args = parser.parse_args()

# Initialize Whisper model early to download to cache.
if args.model_path is not None:
    logging.info("Using provided model...")
    whisper_model = WhisperModel(
        args.model_path, device=args.device, compute_type=mtypes[args.device]
    )
else:
    logging.info("Using default model...")
    whisper_model = WhisperModel(
        args.model_name, device=args.device, compute_type=mtypes[args.device])

if args.vocal_target_path is not None:
    logging.info("Using provided vocal target...")
    vocal_target = args.vocal_target_path
elif args.stemming:
    # Isolate vocals from the rest of the audio
    logging.info("Separating vocals from the rest of the audio...")

    return_code = os.system(
        f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o "temp_outputs"'
    )

    if return_code != 0:
        logging.warning(
            "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
        )
        vocal_target = args.audio
    else:
        vocal_target = os.path.join(
            "temp_outputs", "htdemucs", os.path.basename(args.audio[:-4]), "vocals.wav"
        )
else:
    logging.info("Using original audio file without stemming...")
    vocal_target = args.audio
logging.info("Vocal target: " + vocal_target)

logging.info("Transcribing audio file...")
segments, info = whisper_model.transcribe(
    vocal_target, beam_size=1, word_timestamps=True
)
whisper_results = []
for segment in segments:
    whisper_results.append(segment._asdict())
# clear gpu vram
del whisper_model
torch.cuda.empty_cache()

logging.info("Aligning transcript with audio file...")
if False and info.language in wav2vec2_langs:
    logging.debug("Loading alignment model for language")
    alignment_model, metadata = load_align_model(
        language_code=info.language, device=args.device
    )
    logging.debug("Aligning results")
    result_aligned = whisperx.align(
        whisper_results, alignment_model, metadata, vocal_target, args.device
    )
    word_timestamps = result_aligned["word_segments"]
    # clear gpu vram
    del alignment_model
    torch.cuda.empty_cache()
else:
    word_timestamps = []
    for segment in whisper_results:
        for word in segment["words"]:
            word_timestamps.append({"text": word[2], "start": word[0], "end": word[1]})


# convert audio to mono for NeMo combatibility
logging.info("Converting audio to mono...")
signal, sample_rate = librosa.load(vocal_target, sr=None)
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
soundfile.write(os.path.join(temp_path, "mono_file.wav"), signal, sample_rate, "PCM_24")

# Initialize NeMo MSDD diarization model
logging.info("Initializing NeMo MSDD diarization model...")
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
msdd_model.diarize()

del msdd_model
torch.cuda.empty_cache()

# Reading timestamps <> Speaker Labels mapping
logging.info("Reading timestamps <> Speaker Labels mapping...")
speaker_ts = []
with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

logging.info("Restoring punctuation...")
if info.language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences
    punct_model = PunctuationModel(model="kredor/punctuate-all")

    words_list = list(map(lambda x: x["word"], wsm))

    labled_words = punct_model.predict(words_list)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"

    # We don't want to punctuate U.S.A. with a period. Right?
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

    for word_dict, labeled_tuple in zip(wsm, labled_words):
        word = word_dict["word"]
        if (
            word
            and labeled_tuple[1] in ending_puncts
            and (word[-1] not in model_puncts or is_acronym(word))
        ):
            word += labeled_tuple[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
else:
    logging.warning(
        f'Punctuation restoration is not available for {info.language} language.'
    )

ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

with open(f"{args.audio[:-4]}.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f)

with open(f"{args.audio[:-4]}.srt", "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)

cleanup(temp_path)
