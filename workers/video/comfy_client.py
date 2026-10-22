"""ComfyUI HTTP + WebSocket client (`workers/video/comfy_client.py`).

Handles:
  - Readiness & custom-node probing (`/system_stats`, `/object_info`)
  - Image upload (`/upload/image`)
  - Prompt submission (`/prompt`), WebSocket progress/error/cache monitoring (`/ws?clientId=...`),
    and `/history/{prompt_id}` polling fallback
  - Output video/frame download (`/view`)
  - VRAM unloading (`/free`) and execution cancellation (`/interrupt`)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("m2a.comfy_client")


class ComfyError(RuntimeError):
    """Base error for ComfyUI interactions."""


class ComfyUnavailable(ComfyError):
    """Raised when ComfyUI HTTP endpoint is unreachable."""


class ComfyNodeMissing(ComfyError):
    """Raised when required custom node class_types are missing from `/object_info`."""


class ComfyOOM(ComfyError):
    """Raised when ComfyUI reports CUDA out-of-memory during execution."""


class ComfyTimeout(ComfyError):
    """Raised when execution exceeds timeout_s or is cancelled."""


class ComfyExecutionError(ComfyError):
    """Raised when ComfyUI reports a node execution error."""


class ComfyClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        timeout_s: float = 1800.0,
        poll_interval_s: Optional[float] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        if poll_interval_s is None:
            self.poll_interval_s = 0.05 if http_client is not None else 0.5
        else:
            self.poll_interval_s = float(poll_interval_s)
        self._external_client = http_client

    def _client(self, timeout: float = 30.0) -> httpx.Client:
        if self._external_client is not None:
            return self._external_client
        return httpx.Client(base_url=self.base_url, timeout=timeout)

    def probe(self, required_nodes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Verifies ComfyUI is reachable and all required node classes exist."""
        try:
            client = self._client(timeout=5.0)
            try:
                r_stats = client.get("/system_stats")
                r_stats.raise_for_status()
                stats = r_stats.json()
                r_info = client.get("/object_info")
                r_info.raise_for_status()
                obj_info = r_info.json()
            finally:
                if self._external_client is None:
                    client.close()
        except Exception as exc:
            raise ComfyUnavailable(f"ComfyUI unreachable at {self.base_url}: {exc}") from exc

        devices = stats.get("devices") or []
        vram_gb = 0.0
        if devices:
            vram_total = float(devices[0].get("vram_total", 0))
            vram_gb = round(vram_total / (1024 ** 3), 2) if vram_total > 1024 else vram_total

        missing: List[str] = []
        if required_nodes:
            available = set(obj_info.keys()) if isinstance(obj_info, dict) else set()
            missing = [n for n in required_nodes if n not in available]
            if missing:
                raise ComfyNodeMissing(f"Missing ComfyUI custom nodes: {missing}")

        return {
            "ready": True,
            "vram_gb": vram_gb,
            "missing_nodes": missing,
        }

    def upload_image(self, image_path: Path, overwrite: bool = True) -> str:
        """Uploads a clean plate PNG to ComfyUI's input directory via `POST /upload/image`."""
        image_path = Path(image_path)
        data = {"overwrite": "true" if overwrite else "false", "type": "input"}
        try:
            client = self._client(timeout=30.0)
            try:
                with open(image_path, "rb") as f:
                    files = {"image": (image_path.name, f, "image/png")}
                    resp = client.post("/upload/image", data=data, files=files)
                resp.raise_for_status()
                payload = resp.json()
                return str(payload.get("name") or image_path.name)
            finally:
                if self._external_client is None:
                    client.close()
        except Exception as exc:
            raise ComfyUnavailable(f"Failed to upload image {image_path} to ComfyUI: {exc}") from exc

    def free_memory(self, unload_models: bool = True, free_memory: bool = True) -> None:
        """Requests ComfyUI to unload models and free VRAM (`POST /free`)."""
        client = self._client(timeout=15.0)
        try:
            resp = client.post(
                "/free",
                json={"unload_models": bool(unload_models), "free_memory": bool(free_memory)},
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.debug(f"ComfyUI /free failed or unsupported: {exc}")
        finally:
            if self._external_client is None:
                client.close()

    def interrupt(self) -> None:
        """Interrupts the currently executing ComfyUI prompt (`POST /interrupt`)."""
        client = self._client(timeout=10.0)
        try:
            client.post("/interrupt")
        except Exception as exc:
            logger.debug(f"ComfyUI /interrupt failed: {exc}")
        finally:
            if self._external_client is None:
                client.close()

    def _classify_error(self, err_msg: str) -> ComfyError:
        lower = err_msg.lower()
        if "out of memory" in lower or "cuda error: out of memory" in lower or "oom" in lower:
            return ComfyOOM(err_msg)
        if "missing" in lower and "node" in lower:
            return ComfyNodeMissing(err_msg)
        return ComfyExecutionError(err_msg)

    def submit_and_wait(
        self,
        workflow: Dict[str, Any],
        output_dir: Path,
        client_id: Optional[str] = None,
        cancel_event: Optional[Any] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Submits an API workflow, waits for completion (handling cached/OOM/errors/timeouts),
        and downloads the produced video file into `output_dir`.
        """
        cid = client_id or uuid.uuid4().hex
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        client = self._client(timeout=30.0)
        try:
            try:
                resp = client.post("/prompt", json={"prompt": workflow, "client_id": cid})
            except Exception as exc:
                raise ComfyUnavailable(f"POST /prompt failed: {exc}") from exc

            if resp.status_code >= 400:
                detail = resp.text
                try:
                    j = resp.json()
                    detail = json.dumps(j)
                except Exception:
                    pass
                raise self._classify_error(f"ComfyUI prompt rejected ({resp.status_code}): {detail}")

            prompt_data = resp.json()
            prompt_id = str(prompt_data.get("prompt_id") or "")
            if not prompt_id:
                raise ComfyExecutionError(f"ComfyUI returned empty prompt_id: {prompt_data}")

            deadline = time.monotonic() + self.timeout_s
            peak_vram_gb = 0.0
            cached_execution = False

            while time.monotonic() < deadline:
                if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                    self.interrupt()
                    raise ComfyTimeout(f"Prompt {prompt_id} cancelled by caller")

                h_resp = client.get(f"/history/{prompt_id}")
                if h_resp.status_code == 200:
                    hist_all = h_resp.json()
                    entry = hist_all.get(prompt_id) if isinstance(hist_all, dict) else None
                    if isinstance(entry, dict):
                        status_obj = entry.get("status") or {}
                        status_str = str(status_obj.get("status_str") or "").lower()
                        messages = status_obj.get("messages") or []
                        for m in messages:
                            if isinstance(m, (list, tuple)) and len(m) >= 2:
                                m_type = str(m[0])
                                m_body = m[1] if isinstance(m[1], dict) else {}
                                if m_type == "execution_cached":
                                    cached_execution = True
                                elif m_type == "execution_error":
                                    err_text = str(
                                        m_body.get("exception_message")
                                        or m_body.get("error")
                                        or m_body
                                    )
                                    raise self._classify_error(err_text)

                        if status_str == "error" or status_obj.get("completed") is False and status_obj.get("error"):
                            err_text = str(status_obj.get("error") or status_obj)
                            raise self._classify_error(err_text)

                        outputs = entry.get("outputs") or {}
                        if outputs or status_obj.get("completed") is True or status_str == "success":
                            # Locate output video/animated image file
                            file_info = self._extract_output_file(outputs)
                            if file_info is None:
                                raise ComfyExecutionError(
                                    f"Prompt {prompt_id} completed without video/image outputs: {outputs}"
                                )
                            filename = file_info["filename"]
                            subfolder = file_info.get("subfolder", "")
                            ftype = file_info.get("type", "output")
                            view_resp = client.get(
                                "/view",
                                params={"filename": filename, "subfolder": subfolder, "type": ftype},
                            )
                            view_resp.raise_for_status()
                            dest = output_dir / f"{prompt_id}_{Path(filename).name}"
                            dest.write_bytes(view_resp.content)

                            try:
                                st = client.get("/system_stats").json()
                                devs = st.get("devices") or []
                                if devs:
                                    d0 = devs[0]
                                    v_used = float(d0.get("vram_used") or d0.get("vram_Allocated") or 0.0)
                                    if v_used <= 0.0 and d0.get("vram_total") is not None and d0.get("vram_free") is not None:
                                        v_used = max(0.0, float(d0["vram_total"]) - float(d0["vram_free"]))
                                    peak_vram_gb = round(v_used / (1024 ** 3), 2) if v_used > 1024 else v_used
                            except Exception:
                                pass

                            return dest, {
                                "prompt_id": prompt_id,
                                "cached": cached_execution,
                                "peak_vram_gb": peak_vram_gb,
                                "filename": filename,
                            }

                time.sleep(self.poll_interval_s)

            # Timeout reached: interrupt ComfyUI
            self.interrupt()
            raise ComfyTimeout(f"Prompt {prompt_id} timed out after {self.timeout_s:.1f}s")
        finally:
            if self._external_client is None:
                client.close()

    @staticmethod
    def _extract_output_file(outputs: Dict[str, Any]) -> Optional[Dict[str, str]]:
        for node_out in outputs.values():
            if not isinstance(node_out, dict):
                continue
            for key in ("videos", "gifs", "images", "webp", "webm"):
                items = node_out.get(key)
                if isinstance(items, list) and items:
                    first = items[0]
                    if isinstance(first, dict) and first.get("filename"):
                        return {
                            "filename": str(first["filename"]),
                            "subfolder": str(first.get("subfolder", "")),
                            "type": str(first.get("type", "output")),
                        }
        return None
