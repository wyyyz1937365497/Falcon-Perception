"""
Python 版 iS3 MetadataAPI（参考 metadata.js）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests


class MetadataAPI:
    def __init__(
        self,
        base_url: str,
        prj_id: str,
        headers: Dict[str, str],
        folder_id: Optional[str] = None,
        file_base_url: str = "https://file.is3.net.cn",
        timeout: int = 20,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.prj_id = str(prj_id or "")
        self.headers = dict(headers or {})
        self.folder_id = str(folder_id or "")
        self.file_base_url = (file_base_url or "https://file.is3.net.cn").rstrip("/")
        self.timeout = int(timeout)

    def _auth_headers(self) -> Dict[str, str]:
        access_key = self.headers.get("X-Access-Key", "")
        secret_key = self.headers.get("X-Secret-Key", "")
        if not access_key or not secret_key:
            raise ValueError("iS3 鉴权配置缺失：X-Access-Key 或 X-Secret-Key 为空")
        return {
            "X-Access-Key": access_key,
            "X-Secret-Key": secret_key,
        }

    def _json_headers(self) -> Dict[str, str]:
        return {
            **self._auth_headers(),
            "Content-Type": "application/json",
        }

    def _process_file_url(self, url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        if url.startswith("http://") or url.startswith("https://"):
            return url
        return f"{self.file_base_url}{url}"

    def create_data(self, meta_table_code: str, data: Union[dict, list[dict]]) -> dict:
        return self.insert_data(meta_table_code, data)

    def insert_data(self, meta_table_code: str, data: Union[dict, list[dict]]) -> dict:
        payload = {
            "prjId": self.prj_id,
            "metaTableCode": meta_table_code,
            "data": data if isinstance(data, list) else [data],
        }
        resp = requests.post(
            f"{self.base_url}/data-main/operation/addData",
            headers=self._json_headers(),
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_file_list(self, folder_id: Optional[str] = None, page_num: int = 1, page_size: int = 13) -> dict:
        fid = str(folder_id or self.folder_id)
        if not fid:
            raise ValueError("folderId 不能为空")
        url = (
            f"{self.base_url}/system/material/list"
            f"?pageNum={page_num}&pageSize={page_size}&folderId={fid}"
        )
        resp = requests.get(url, headers=self._auth_headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def upload_file(self, file_path: str, folder_id: Optional[str] = None) -> dict:
        fid = str(folder_id or self.folder_id)
        if not fid:
            raise ValueError("folderId 不能为空")

        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        url = f"{self.base_url}/system/material/uploadFile/{fid}"
        with path.open("rb") as fp:
            files = {"file": (path.name, fp)}
            resp = requests.post(
                url,
                headers=self._auth_headers(),
                files=files,
                timeout=self.timeout,
            )

        resp.raise_for_status()
        return resp.json()

    def upload_file_get_access_url(self, file_path: str, folder_id: Optional[str] = None) -> Optional[str]:
        result = self.upload_file(file_path, folder_id)
        if not isinstance(result, dict) or not result.get("success"):
            return None
        data = result.get("data") or {}
        return self._process_file_url(data.get("url"))

    def query_data(
        self,
        meta_table_code: str,
        page_number: int = 0,
        page_size: int = 0,
        where_conditions: Optional[list[dict]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        desc: bool = True,
    ) -> dict:
        """查询数据"""
        payload = {
            "pageNumber": page_number,
            "pageSize": page_size,
            "prjId": self.prj_id,
            "metaTableCode": meta_table_code,
            "desc": desc,
            "whereConditions": where_conditions or [],
        }
        if start_time:
            payload["startTime"] = start_time
        if end_time:
            payload["endTime"] = end_time

        resp = requests.post(
            f"{self.base_url}/data-main/operation/getDataByCondition",
            headers=self._json_headers(),
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def update_data(
        self,
        meta_table_code: str,
        data: Union[dict, list[dict]],
        where_conditions: list[dict],
    ) -> dict:
        """更新数据"""
        payload = {
            "prjId": self.prj_id,
            "metaTableCode": meta_table_code,
            "data": data if isinstance(data, list) else [data],
            "whereConditions": where_conditions,
        }
        resp = requests.post(
            f"{self.base_url}/data-main/operation/updateDataByCondition",
            headers=self._json_headers(),
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def delete_data(
        self,
        meta_table_code: str,
        where_conditions: list[dict],
    ) -> dict:
        """删除数据"""
        payload = {
            "prjId": self.prj_id,
            "metaTableCode": meta_table_code,
            "whereConditions": where_conditions,
        }
        resp = requests.post(
            f"{self.base_url}/data-main/operation/deleteDataByCondition",
            headers=self._json_headers(),
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()
