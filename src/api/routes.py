from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING

from aiohttp import web
from aiohttp.web import Response

from utils.cors import add_cors_routes
from utils.limiter import Limiter
from utils.whisper import transcribe_file, transcribe_bytes

if TYPE_CHECKING:
  from utils.extra_request import Request

with open("config.toml") as f:
  config = tomllib.loads(f.read())
  frontend_version = config["pages"]["frontend_version"]
  exempt_ips = config["srv"]["ratelimit_exempt"]
  api_version = config["srv"]["api_version"]

limiter = Limiter(exempt_ips=exempt_ips, use_auth=False)
routes = web.RouteTableDef()


@routes.get("/srv/get/")
@limiter.limit("60/m")
async def get_lp_get(request: Request) -> Response:
  packet = {
    "frontend_version": frontend_version,
    "api_version": api_version,
  }

  if request.app.POSTGRES_ENABLED:
    database_size_record = await request.conn.fetchrow(
      "SELECT pg_size_pretty ( pg_database_size ( current_database() ) );"
    )
    packet["db_size"] = database_size_record.get("pg_size_pretty", "-1 kB")

  return web.json_response(packet)


@routes.post("/whisper/transcribe/file/")
@limiter.limit("6/m")
async def post_whisper_transcribe_file(request: Request) -> Response:
  data = await request.read()

  query = request.query

  detailed = query.get("detailed", "false").lower() == "true"
  vad = query.get("vad", "false").lower() == "true"
  vad_options = None
  try:
    threshold = float(query.get("vad_threshold", 0.5))
    min_speech_ms = float(query.get("vad_min_speech", 250))
    max_speech_s = float(query.get("vad_max_speech", float("inf")))
    min_silence_ms = float(query.get("vad_min_silence", 2000))
    speech_pad = float(query.get("vad_speech_pad", 400))
    vad_options = {
      "threshold": threshold,
      "min_speech_duration_ms": min_speech_ms,
      "max_speech_duration_s": max_speech_s,
      "min_silence_duration_ms": min_silence_ms,
      "speech_pad_ms": speech_pad,
    }
  except ValueError:
    return Response(status=400, text="failed converting vad options")

  try:
    result = await transcribe_file(data, use_vad=vad, vad_options=vad_options)
    if not detailed:
      return Response(text=result.full_text)
    else:
      segments = []
      for segment in result.segments:
        words = []
        try:
          for word in segment.words:
            words.append(
              {
                "start": word.start,
                "end": word.end,
                "word": word.word,
                "probability": word.probability,
              }
            )
        except Exception:
          pass

        segments.append(
          {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": words,
          }
        )

      packet = {
        "full_text": result.full_text,
        "language": result.language,
        "language_probability": result.language_prob,
        "segments": segments,
        "duration": result.info.duration,
      }
      return web.json_response(packet)
  except Exception:
    request.LOG.exception("Failed transcription!")
    return Response(status=500)


@routes.post("/whisper/transcribe/raw/")
@limiter.limit("6/m")
async def post_whisper_transcribe_raw(request: Request) -> Response:
  data = await request.read()

  query = request.query

  detailed = query.get("detailed", "false").lower() == "true"
  vad = query.get("vad", "false").lower() == "true"
  vad_options = None
  try:
    threshold = float(query.get("vad_threshold", 0.5))
    min_speech_ms = float(query.get("vad_min_speech", 250))
    max_speech_s = float(query.get("vad_max_speech", float("inf")))
    min_silence_ms = float(query.get("vad_min_silence", 2000))
    speech_pad = float(query.get("vad_speech_pad", 400))
    vad_options = {
      "threshold": threshold,
      "min_speech_duration_ms": min_speech_ms,
      "max_speech_duration_s": max_speech_s,
      "min_silence_duration_ms": min_silence_ms,
      "speech_pad_ms": speech_pad,
    }
  except ValueError:
    return Response(status=400, text="failed converting vad options")

  try:
    result = await transcribe_bytes(data, use_vad=vad, vad_options=vad_options)
    if not detailed:
      return Response(text=result.full_text)
    else:
      segments = []
      for segment in result.segments:
        words = []
        try:
          for word in segment.words:
            words.append(
              {
                "start": word.start,
                "end": word.end,
                "word": word.word,
                "probability": word.probability,
              }
            )
        except Exception:
          pass

        segments.append(
          {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": words,
          }
        )

      packet = {
        "full_text": result.full_text,
        "language": result.language,
        "language_probability": result.language_prob,
        "segments": segments,
        "duration": result.info.duration,
      }
      return web.json_response(packet)
  except Exception:
    request.LOG.exception("Failed transcription!")
    return Response(status=500)


async def setup(app: web.Application) -> None:
  for route in routes:
    app.LOG.info(f"  ↳ {route}")
  app.add_routes(routes)
  add_cors_routes(routes, app)
