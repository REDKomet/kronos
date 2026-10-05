"""External Python service. Do NOT execute inside Base44. Single-replica demo."""
import os
import time
import uuid
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal
import numpy as np
import pandas as pd

import exchange_calendars as xcals
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field, ConfigDict, model_validator
from openai import OpenAI
STOCKS = {"NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM"}
jobs = {}
lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)
predictor = None

class Candle(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    time: int = Field(gt=0)
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def consistent(self):
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("Invalid OHLC candle")
        return self

class Forecast(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    symbol: str
    interval: Literal["daily", "15m"]
    horizon: Literal[5, 10, 20]
    samples: Literal[5, 10, 20]
    candles: list[Candle] = Field(min_length=64, max_length=400)

    @model_validator(mode="after")
    def ordered(self):
        if self.symbol not in STOCKS:
            raise ValueError("Unsupported US stock")
        times = [c.time for c in self.candles]
        if any(b <= a for a, b in zip(times, times[1:])) or times[-1] > time.time()*1000:
            raise ValueError("Invalid historical timestamps")
        return self

def authenticated(authorization: str = Header(default="")):
    expected = os.environ.get("KRONOS_SERVICE_KEY", "")
    if not expected or not secrets.compare_digest(authorization, "Bearer " + expected):
        raise HTTPException(401, "Unauthorized")

def future_times(last, interval, horizon):
    start = pd.Timestamp(last, unit="ms", tz="UTC").tz_convert("America/New_York").normalize().tz_localize(None)
    calendar = xcals.get_calendar("XNYS")
    schedule = calendar.schedule.loc[str(start.date()):str((start + pd.Timedelta(days=90)).date())]
    candidates = []
    for _, row in schedule.iterrows():
        opened = pd.Timestamp(row["open"])
        closed = pd.Timestamp(row["close"])
        if opened.tzinfo is None:
            opened = opened.tz_localize("UTC")
            closed = closed.tz_localize("UTC")
        values = [opened] if interval == "daily" else pd.date_range(opened, closed, freq="15min", inclusive="left")
        for stamp in values:
            millis = int(stamp.timestamp()*1000)
            if millis > last:
                candidates.append(millis)
            if len(candidates) == horizon:
                return candidates
    raise ValueError("Unable to build a future exchange calendar")

def generate(job_id, request):
    started = time.monotonic()
    try:
        with lock:
            jobs[job_id]["status"] = "running"
        
        # Format your candle data for the API prompt
        prompt_data = f"Analyze these {len(request.candles)} candles and predict the next {request.horizon} prices."
        
        response = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": prompt_data}]
        )
        
        with lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = response.choices[0].message.content
            
    except Exception as e:
        with lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)

@asynccontextmanager


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

@app.get("/health")
def health():
    return {"ready": True, "model": "meta/llama-3.3-70b-instruct"}
   

@app.post("/jobs",dependencies=[Depends(authenticated)])
def start(request: Forecast):
    with lock:
        for key in list(jobs):
            if time.time()-jobs[key]["created"] > 3600 and jobs[key]["status"] in ("completed","failed"):
                del jobs[key]
        pending = [j for j in jobs.values() if j["status"] in ("queued","running")]
        if len(pending) >= 4 or any(j["user_id"] == request.user_id for j in pending):
            raise HTTPException(429,"A forecast is already running or the service is busy")
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"user_id":request.user_id,"status":"queued","created":time.time()}
    executor.submit(generate,job_id,request)
    return {"job_id":job_id}

@app.get("/jobs/{job_id}",dependencies=[Depends(authenticated)])
def status(job_id: str,user_id: str):
    with lock:
        job = jobs.get(job_id)
        if not job or job["user_id"] != user_id:
            raise HTTPException(404,"Job unavailable or expired")
        return {key:job[key] for key in ("status","result","error") if key in job}
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
