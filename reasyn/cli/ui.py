from __future__ import annotations

import click
import uvicorn


@click.command(context_settings={"show_default": True})
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8765, type=int)
@click.option("--reload", is_flag=True, help="Reload the local UI server on code changes.")
def main(host: str, port: int, reload: bool) -> None:
    """Run the local ReaSyn Modal endpoint UI."""
    click.echo(f"Starting ReaSyn UI at http://{host}:{port}")
    uvicorn.run("reasyn.ui_server:app", host=host, port=port, reload=reload)
