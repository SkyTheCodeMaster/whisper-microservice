from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING

from aiohttp import web
from aiohttp.web import Response

from utils.authenticate import Approval, Key, authenticate, get_project_status
from utils.cors import add_cors_routes
from utils.limiter import Limiter
from utils.whisper import transcribe_file, transcribe_bytes

if TYPE_CHECKING:
  from utils.authenticate import User
  from utils.extra_request import Request

with open("config.toml") as f:
  config = tomllib.loads(f.read())
  frontend_version = config["pages"]["frontend_version"]
  exempt_ips = config["srv"]["ratelimit_exempt"]
  api_version = config["srv"]["api_version"]

limiter = Limiter(exempt_ips=exempt_ips)
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
@limiter.limit("1/10s")
async def post_whisper_transcribe_file(request: Request) -> Response:
  auth = await authenticate(request, cs=request.session)

  if isinstance(auth, Response):
    return auth
  else:
    if isinstance(auth, Key):
      # this means its a Key
      user: User = auth.user
    else:
      user: User = auth

    status = await get_project_status(user, "Whisper", cs=request.session)
    if status != Approval.APPROVED:
      return Response(
        status=401,
        text="please apply for project at https://auth.skystuff.cc/projects#Whisper",
      )

  data = await request.read()

  detailed = request.query.get("detailed", "false").lower() == "true"

  try:
    result = await transcribe_file(data)
    if not detailed:
      return Response(text=result.full_text)
    else:
      segments = []
      for segment in result.segments:
        words = []
        for word in segment.words:
          words.append({
            "start": word.start,
            "end": word.end,
            "word": word.word,
            "probability": word.probability
          })
        
        segments.append({
          "start": segment.start,
          "end": segment.end,
          "text": segment.text,
          "words": words,
        })

      packet = {
        "full_text": result.full_text,
        "language": result.language,
        "language_probability": result.language_prob,
        "segments": segments,
        "duration": result.info.duration
      }
      return web.json_response(packet)
  except Exception:
    request.LOG.exception("Failed transcription!")
    return Response(status=500)

@routes.post("/whisper/transcribe/raw/")
@limiter.limit("1/10s")
async def post_whisper_transcribe_raw(request: Request) -> Response:
  auth = await authenticate(request, cs=request.session)

  if isinstance(auth, Response):
    return auth
  else:
    if isinstance(auth, Key):
      # this means its a Key
      user: User = auth.user
    else:
      user: User = auth

    status = await get_project_status(user, "Whisper", cs=request.session)
    if status != Approval.APPROVED:
      return Response(
        status=401,
        text="please apply for project at https://auth.skystuff.cc/projects#Whisper",
      )

  data = await request.read()

  detailed = request.query.get("detailed", "false").lower() == "true"

  try:
    result = await transcribe_bytes(data)
    if not detailed:
      return Response(text=result.full_text)
    else:
      segments = []
      for segment in result.segments:
        words = []
        for word in segment.words:
          words.append({
            "start": word.start,
            "end": word.end,
            "word": word.word,
            "probability": word.probability
          })
        
        segments.append({
          "start": segment.start,
          "end": segment.end,
          "text": segment.text,
          "words": words,
        })

      packet = {
        "full_text": result.full_text,
        "language": result.language,
        "language_probability": result.language_prob,
        "segments": segments,
        "duration": result.info.duration
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
