"""Hot-reload watcher for config.yaml.

watchdog runs in a background thread. When it sees a modification event,
it schedules an async reload callback on the main asyncio loop via
`run_coroutine_threadsafe`. The async callback re-reads the config,
validates it, and (if valid) applies the change. A bad reload logs a
warning and keeps the previous config.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable

from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .config import ConfigError, RouterConfig, load_config


logger = logging.getLogger(__name__)


ReloadCallback = Callable[[RouterConfig], Awaitable[None]]


class _Handler(FileSystemEventHandler):
    def __init__(
        self,
        config_path: Path,
        loop: asyncio.AbstractEventLoop,
        on_reload: ReloadCallback,
    ):
        self._path = config_path.resolve()
        self._loop = loop
        self._on_reload = on_reload

    def on_modified(self, event: FileModifiedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        if Path(event.src_path).resolve() != self._path:
            return
        asyncio.run_coroutine_threadsafe(self._reload(), self._loop)

    async def _reload(self) -> None:
        try:
            cfg = load_config(self._path)
        except ConfigError as e:
            logger.warning("Config reload rejected (keeping previous): %s", e)
            return
        except Exception as e:  # noqa: BLE001
            logger.warning("Config reload failed (keeping previous): %s", e)
            return
        try:
            await self._on_reload(cfg)
        except Exception as e:  # noqa: BLE001
            logger.exception("Reload callback failed: %s", e)


class ConfigWatcher:
    def __init__(
        self,
        config_path: Path,
        loop: asyncio.AbstractEventLoop,
        on_reload: ReloadCallback,
    ):
        self._observer = Observer()
        handler = _Handler(config_path, loop, on_reload)
        # Watch the parent directory; we filter to the specific file in the handler.
        self._observer.schedule(handler, str(config_path.parent), recursive=False)

    def start(self) -> None:
        self._observer.start()

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join(timeout=2.0)
