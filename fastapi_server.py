"""
FastAPI 测试服务端
接收 RTSP 检测服务上报的数据和图片，保存到本地 fastapi_test 文件夹。

启动命令:
    cd F:\TJ\Falcon-Perception
    python fastapi_server.py
"""

import os
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 保存根目录
SAVE_DIR = Path(__file__).resolve().parent / "fastapi_test"
IMAGES_DIR = SAVE_DIR / "images"
DETAILS_DIR = SAVE_DIR / "details"
SIMPLE_DIR = SAVE_DIR / "simple"

app = FastAPI(title="Falcon-Perception Test Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectionDetail(BaseModel):
    camera_code: Optional[str] = None
    camera_time: Optional[str] = None
    camera_img: Optional[str] = None
    result_json: Optional[str] = None
    result_num: Optional[int] = 0
    region_id: Optional[str] = None


class DetectionSimple(BaseModel):
    camera_code: Optional[str] = None
    camera_time: Optional[str] = None
    camera_result: Optional[int] = 0
    region_id: Optional[str] = None


@app.on_event("startup")
def startup():
    for d in [IMAGES_DIR, DETAILS_DIR, SIMPLE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"FastAPI 测试服务启动，数据保存到: {SAVE_DIR}")


@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


@app.post("/api/detection/detail")
def post_detail(data: DetectionDetail):
    ts = data.camera_time or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    region = data.region_id or "default"
    safe_ts = ts.replace(":", "-").replace(" ", "_")

    filename = f"{safe_ts}_{region}.json"
    filepath = DETAILS_DIR / filename

    save_data = data.dict()
    save_data["received_at"] = datetime.now().isoformat()

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    logger.info(f"[detail] {region} | {ts} | count={data.result_num} → {filename}")
    return {"status": "ok", "saved": str(filepath)}


@app.post("/api/detection/simple")
def post_simple(data: DetectionSimple):
    ts = data.camera_time or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    region = data.region_id or "default"
    safe_ts = ts.replace(":", "-").replace(" ", "_")

    filename = f"{safe_ts}_{region}.json"
    filepath = SIMPLE_DIR / filename

    save_data = data.dict()
    save_data["received_at"] = datetime.now().isoformat()

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    logger.info(f"[simple] {region} | {ts} | result={data.camera_result} → {filename}")
    return {"status": "ok", "saved": str(filepath)}


@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    safe_name = file.filename.replace(":", "-").replace(" ", "_")
    ts_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts_prefix}_{safe_name}"
    filepath = IMAGES_DIR / filename

    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    logger.info(f"[image] {filename} ({len(content)} bytes)")
    return {"status": "ok", "saved": str(filepath), "size": len(content)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
