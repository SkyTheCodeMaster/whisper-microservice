from __future__ import annotations

import asyncio
import random
import string
import time
import tomllib
import numpy
from typing import TYPE_CHECKING

import aiofiles
import aiofiles.os
from faster_whisper import WhisperModel

if TYPE_CHECKING:
  from faster_whisper.transcribe import Segment, TranscriptionInfo

with open("config.toml") as f:
  config = tomllib.loads(f.read())

MODEL_SIZE = config["model"]["model"]
DEVICE = config["model"]["device"]

model = WhisperModel(
  MODEL_SIZE,
  device=DEVICE,
)

model_lock = asyncio.Lock()

class TranscriptionResult:
  segments: list[Segment]
  info: TranscriptionInfo
  full_text: str
  language: str
  language_prob: float

  def __init__(self, segments: list[Segment], info: TranscriptionInfo) -> None:
    self.segments = segments
    self.info = info
    self.full_text = " ".join(segment.text for segment in segments)
    self.language = info.language
    self.language_prob = info.language_probability


def _transcribe_file(file_path: str) -> TranscriptionResult:
  "Take in PCM WAV 16 bit bytes, return transcribed text"
  segments, info = model.transcribe(file_path)

  segments = list(segments)
  result = TranscriptionResult(segments, info)
  return result


async def transcribe_file(audio_bytes: bytes) -> TranscriptionResult:
  loop = asyncio.get_running_loop()
  start = time.time()
  def t(): return round(time.time()-start, 4)
  print(f"[{t()}] Starting conversion...")
  file_path = await convert_to_wav(audio_bytes)
  print(f"[{t()}] Converted to wav, waiting for lock...")
  async with model_lock:
    print(f"[{t()}] Lock acquired, transcribing...")
    result = await loop.run_in_executor(None, _transcribe_file, file_path)
  print(f"[{t()}] Finished transcription.")
  await aiofiles.os.remove(file_path)
  print(f"[{t()}] Deleted file..")
  return result

def _transcribe_bytes(pcm_bytes: bytes) -> TranscriptionResult:
  audio_array = numpy.frombuffer(pcm_bytes, numpy.int16).astype(numpy.float32) / 32768.0

  segments, info = model.transcribe(audio_array)

  segments = list(segments)
  result = TranscriptionResult(segments, info)
  return result

async def transcribe_bytes(pcm_bytes: bytes) -> TranscriptionResult:
  loop = asyncio.get_running_loop()
  start = time.time()
  def t(): return round(time.time()-start, 4)
  print(f"[{t()}] Waiting for lock...")
  async with model_lock:
    print(f"[{t()}] Lock acquired, transcribing...")
    result = await loop.run_in_executor(None, _transcribe_bytes, pcm_bytes)
  print(f"[{t()}] Finished transcription.")
  return result


async def convert_to_wav(data: bytes) -> str:
  "Convert an audio file to wav by saving it as a temporary file and using ffmpeg to convert it."
  pool: str = string.ascii_letters + string.digits
  job_id = "".join(random.choices(pool, k=32))

  await aiofiles.os.makedirs("/tmp/audioconversion/", exist_ok=True)

  async with aiofiles.open(f"/tmp/audioconversion/{job_id}.src", "wb") as f:
    await f.write(data)

  proc = await asyncio.create_subprocess_shell(
    f"ffmpeg -i /tmp/audioconversion/{job_id}.src -vn -acodec pcm_s16le -ar 16000 -ac 1 /tmp/audioconversion/{job_id}.wav"
  )

  returncode = await proc.wait()

  if returncode != 0:
    raise Exception("Failed to convert audio file.")

  await aiofiles.os.remove(f"/tmp/audioconversion/{job_id}.src")

  return f"/tmp/audioconversion/{job_id}.wav"
