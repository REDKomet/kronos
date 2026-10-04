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
import torch
import exchange_calendars as xcals
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field, ConfigDict, model_validator
from model import Kronos, KronosTokenizer, KronosPredictor

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
        frame = pd.DataFrame([c.model_dump() for c in request.candles])
        frame["amount"] = frame["close"]*frame["volume"]
        times = future_times(request.candles[-1].time, request.interval, request.horizon)
        x_dates = pd.to_datetime(frame["time"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        y_dates = pd.Series(pd.to_datetime(times, unit="ms", utc=True).tz_convert("America/New_York").tz_localize(None))
        paths = []
        for _ in range(request.samples):
            if time.monotonic()-started > 180:
                raise TimeoutError("Inference time limit exceeded")
            with torch.inference_mode():
                predicted = predictor.predict(df=frame[["open","high","low","close","volume","amount"]], x_timestamp=x_dates, y_timestamp=y_dates, pred_len=request.horizon, T=1.0, top_p=0.9, sample_count=1, verbose=False)
            path = predicted[["open","high","low","close"]].to_numpy(dtype=float)
            if path.shape != (request.horizon,4) or not np.isfinite(path).all() or (path <= 0).any():
                raise ValueError("Model produced invalid prices")
            # Enforce valid OHLC envelopes for every decoded path.
            path[:,1] = np.max(path, axis=1)
            path[:,2] = np.min(path, axis=1)
            paths.append(path)
        if time.monotonic()-started > 180:
            raise TimeoutError("Inference time limit exceeded")
        array = np.stack(paths)
        median = np.median(array,axis=0)
        lower, upper = np.quantile(array[:,:,3], [.05,.95],axis=0)
        candles = [{"time": t, "open": float(median[i,0]), "high": float(median[i,1]), "low": float(median[i,2]), "close": float(median[i,3]), "lower": float(lower[i]), "upper": float(upper[i])} for i,t in enumerate(times)]
        result = {"candles":candles, "samples":request.samples, "model":"Kronos-small", "generated_at":datetime.now(timezone.utc).isoformat(), "generation_seconds":round(time.monotonic()-started,3), "amount_method":"close_times_volume", "calendar":"XNYS", "percentiles":[5,95]}
        with lock:
            jobs[job_id].update(status="completed",result=result)
    except Exception as error:
        print(f"Kronos job {job_id} failed: {type(error).__name__}",flush=True)
        with lock:
            jobs[job_id].update(status="failed",error="Kronos could not generate valid scenarios within the time limit. Please retry.")

@asynccontextmanager
async def lifespan(app):
    global predictor
    if not os.environ.get("KRONOS_SERVICE_KEY"):
        raise RuntimeError("Set KRONOS_SERVICE_KEY before starting")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval()
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small").eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(model,tokenizer,device=device,max_context=512)
    yield
    executor.shutdown(wait=False,cancel_futures=True)

app = FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)

@app.get("/health")
def health():
    return {"ready":predictor is not None,"model":"Kronos-small"}

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
