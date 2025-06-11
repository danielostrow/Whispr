from pathlib import Path
import json

import dash
from dash import dcc, html, Input, Output
import plotly.express as px

ASSETS_URL_PATH = "/assets/"


def build_app(metadata_path: Path):
    meta = json.loads(Path(metadata_path).read_text())
    df = {
        "id": [spk["id"] for spk in meta["speakers"]],
        "x": [spk["location"][0] for spk in meta["speakers"]],
        "y": [spk["location"][1] for spk in meta["speakers"]],
        "audio": [spk["audio_file"] for spk in meta["speakers"]],
    }

    fig = px.scatter(df, x="x", y="y", text="id", size_max=15)
    fig.update_traces(marker=dict(size=20))
    fig.update_layout(clickmode="event+select", title="Whispr Speaker Map")

    app = dash.Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Graph(id="graph", figure=fig, style={"height": "70vh"}),
            html.Audio(id="player", controls=True, src=""),
        ]
    )

    @app.callback(Output("player", "src"), Input("graph", "clickData"))
    def play_audio(clickData):
        if clickData and "points" in clickData:
            point = clickData["points"][0]
            idx = point["pointIndex"]
            file_rel = df["audio"][idx]
            # Dash serves assets under /assets
            return f"{ASSETS_URL_PATH}{file_rel}"
        return dash.no_update

    return app


if __name__ == "__main__":
    import argparse
    from ..config import Config

    parser = argparse.ArgumentParser(description="Launch Whispr Dash UI")
    parser.add_argument("metadata", type=Path, help="Path to metadata.json")
    args = parser.parse_args()

    meta_path = Path(args.metadata)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    # Copy audio files to assets directory for Dash static serving
    assets_dir = Path(__file__).resolve().parent / "assets"
    assets_dir.mkdir(exist_ok=True)
    meta = json.loads(meta_path.read_text())
    for spk in meta["speakers"]:
        src = meta_path.parent / spk["audio_file"]
        dst = assets_dir / spk["audio_file"]
        if not dst.exists():
            dst.write_bytes(src.read_bytes())

    app = build_app(meta_path)
    app.run(debug=True) 