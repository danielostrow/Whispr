import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

ASSETS_URL_PATH = "/assets/"


def prepare_data(metadata: Dict[str, Any]) -> pd.DataFrame:
    """Extract speaker data from metadata and return a Pandas DataFrame."""
    records = []
    for spk in metadata["speakers"]:
        # Handle both 2D and 3D locations
        location = spk["location"]
        x = location[0]
        y = location[1]
        z = location[2] if len(location) > 2 else 0.0

        records.append(
            {
                "id": spk["id"],
                "x": x,
                "y": y,
                "z": z,
                "audio_file": spk["audio_file"],
            }
        )
    return pd.DataFrame(records)


def create_speaker_map(df: pd.DataFrame) -> go.Figure:
    """Create a Plotly 3D scatter plot of speaker locations."""
    # Check if we have meaningful Z values
    has_z_dimension = df["z"].abs().sum() > 0.01

    if has_z_dimension:
        # Create 3D scatter plot
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            text="id",
            hover_name="id",
            hover_data={"x": False, "y": False, "z": False, "audio_file": True},
            size_max=20,
        )
        fig.update_traces(marker=dict(size=10, symbol="circle"))
        fig.update_layout(
            scene=dict(
                xaxis_title="X Coordinate",
                yaxis_title="Y Coordinate",
                zaxis_title="Z Coordinate (Height)",
            ),
            clickmode="event+select",
            title="Whispr 3D Speaker Map",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
    else:
        # Fall back to 2D scatter plot if Z values are all zero
        fig = px.scatter(
            df,
            x="x",
            y="y",
            text="id",
            hover_name="id",
            hover_data={"x": False, "y": False, "audio_file": True},
            size_max=20,
        )
        fig.update_traces(marker=dict(size=25, symbol="circle"))
        fig.update_layout(
            clickmode="event+select",
            title="Whispr Speaker Map",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )

    # Add room outline
    if has_z_dimension:
        # Add a transparent cube to represent the room (assuming 5x5x3m room)
        room_width, room_length, room_height = 5.0, 5.0, 3.0

        # Create the 8 corners of the cube
        x = [0, room_width, room_width, 0, 0, room_width, room_width, 0]
        y = [0, 0, room_length, room_length, 0, 0, room_length, room_length]
        z = [0, 0, 0, 0, room_height, room_height, room_height, room_height]

        # Define the 12 edges of the cube
        i = [0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 4, 5, 5, 6, 6, 7]
        j = [1, 3, 4, 2, 5, 3, 6, 7, 5, 7, 0, 6, 1, 7, 2, 3]

        # Create a mesh3d trace for the room
        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=[],
                opacity=0.1,
                color="lightblue",
                hoverinfo="none",
            )
        )

    return fig


def create_layout(fig: go.Figure) -> html.Div:
    """Create the Dash application layout."""
    return html.Div(
        className="container",
        children=[
            html.H1("Whispr Speaker Diarization"),
            dcc.Graph(
                id="speaker-map",
                figure=fig,
                className="speaker-map",
                config={"displayModeBar": True},
            ),
            html.Audio(id="audio-player", controls=True, src=""),
            html.Div(id="speaker-info", className="speaker-info"),
        ],
    )


def build_app(metadata_path: Path) -> Dash:
    """Build the full Dash application."""
    metadata = json.loads(metadata_path.read_text())
    df = prepare_data(metadata)
    fig = create_speaker_map(df)

    app = Dash(__name__, assets_url_path=ASSETS_URL_PATH)
    app.layout = create_layout(fig)

    @app.callback(
        [Output("audio-player", "src"), Output("speaker-info", "children")],
        Input("speaker-map", "clickData"),
    )
    def update_audio_and_info(clickData):
        if clickData and "points" in clickData:
            point_data = clickData["points"][0]
            audio_file = point_data["customdata"][0]
            speaker_id = point_data["hovertext"]

            # Get coordinates for display
            x = point_data["x"]
            y = point_data["y"]
            z = point_data.get("z", 0)  # Handle both 2D and 3D plots

            info_text = [
                html.H3(f"Speaker: {speaker_id}"),
                html.P(f"Location: X={x:.2f}m, Y={y:.2f}m, Z={z:.2f}m"),
                html.P(f"Audio: {audio_file}"),
            ]

            return f"{ASSETS_URL_PATH}{audio_file}", info_text
        return dash.no_update, dash.no_update

    return app


def prepare_assets(metadata_path: Path):
    """Copy audio files into the assets directory for serving."""
    assets_dir = Path(__file__).resolve().parent / "assets"
    assets_dir.mkdir(exist_ok=True)
    metadata = json.loads(metadata_path.read_text())

    for spk in metadata["speakers"]:
        src_path = metadata_path.parent / spk["audio_file"]
        dst_path = assets_dir / spk["audio_file"]
        if not dst_path.exists():
            shutil.copy(src_path, dst_path)
    print(f"Copied {len(metadata['speakers'])} audio files to {assets_dir}")


def main():
    """Parse arguments and run the Dash app."""
    parser = argparse.ArgumentParser(description="Launch the Whispr Dash UI.")
    parser.add_argument(
        "metadata",
        type=Path,
        help="Path to the metadata.json file generated by the pipeline.",
    )
    args = parser.parse_args()

    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")

    prepare_assets(args.metadata)
    app = build_app(args.metadata)
    app.run(debug=True)


if __name__ == "__main__":
    main()
