import time
import json
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import streamlit as st
import streamlit.components.v1 as components
import folium
from folium import plugins

st.set_page_config(page_title="Oil Spill Trajectory Viewer", layout="wide")

default_state = {
    "selected_slick": None,
    "current_frame": 0,
    "playing": False,
    "frames_cache": {},
    "slicks_metadata": [],
    "trajectory_data": {},
    "fps_setting": 12,
    "png_bytes": None,
}
for k, v in default_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

def load_trajectory_from_uploaded_file(uploaded_file):
    """Load trajectory JSON from uploaded file."""
    try:
        content = uploaded_file.read()
        data = json.loads(content)
        return data
    except Exception:
        return None

def compute_trajectory_bounds(trajectory_data, padding_ratio: float = 0.1):
    """Compute lon/lat bounds (with padding) for all points."""
    trajectories = trajectory_data.get("trajectory", [])
    if not trajectories:
        return None

    all_points = []
    for traj in trajectories:
        all_points.extend(traj.get("points", []))

    if not all_points:
        return None

    lons = np.array([p[0] for p in all_points], dtype=np.float32)
    lats = np.array([p[1] for p in all_points], dtype=np.float32)

    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())

    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    lon_min -= lon_range * padding_ratio
    lon_max += lon_range * padding_ratio
    lat_min -= lat_range * padding_ratio
    lat_max += lat_range * padding_ratio

    return lon_min, lat_min, lon_max, lat_max

def extract_slick_metadata_from_upload(uploaded_file):
    """Extract metadata from uploaded file."""
    try:
        data = load_trajectory_from_uploaded_file(uploaded_file)
        if not data:
            return None

        trajectories = data.get("trajectory", [])
        if not trajectories:
            return None

        all_points = []
        for traj in trajectories:
            all_points.extend(traj.get("points", []))

        if not all_points:
            return None

        lons = np.array([p[0] for p in all_points], dtype=np.float32)
        lats = np.array([p[1] for p in all_points], dtype=np.float32)

        centroid_lat = float(lats.mean())
        centroid_lon = float(lons.mean())

        bounds = {
            "lon_min": float(lons.min()),
            "lon_max": float(lons.max()),
            "lat_min": float(lats.min()),
            "lat_max": float(lats.max()),
        }

        file_id = uploaded_file.name.replace('.json', '')

        return {
            "id": file_id,
            "file": uploaded_file.name,
            "centroid": [centroid_lat, centroid_lon],
            "bounds": bounds,
            "num_frames": len(trajectories),
            "time_start": trajectories[0].get("datetime", "N/A"),
            "time_end": trajectories[-1].get("datetime", "N/A"),
            "data": data
        }
    except Exception:
        return None

def apply_colormap_rgba(data: np.ndarray) -> np.ndarray:
    """Vectorized colormap."""
    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    mask0 = data == 0
    rgba[mask0] = [0, 0, 0, 0]

    mask1 = (data > 0) & (data < 25)
    rgba[mask1] = [0, 0, 255, 200]

    mask2 = (data >= 25) & (data < 75)
    rgba[mask2] = [0, 255, 255, 220]

    mask3 = (data >= 75) & (data < 150)
    rgba[mask3] = [255, 255, 0, 230]

    mask4 = (data >= 150) & (data < 200)
    rgba[mask4] = [255, 128, 0, 240]

    mask5 = data >= 200
    rgba[mask5] = [255, 0, 0, 255]

    return rgba

def generate_particle_count_frame(
    trajectory_data,
    frame_idx: int,
    width: int,
    height: int,
    apply_smoothing: bool = True,
    sigma: float = 6.0,
):
    """Generate density frame."""
    trajectories = trajectory_data.get("trajectory", [])
    if frame_idx >= len(trajectories) or frame_idx < 0:
        return None, None

    if "_bounds" not in trajectory_data:
        bounds = compute_trajectory_bounds(trajectory_data)
        if bounds is None:
            return None, None
        trajectory_data["_bounds"] = bounds
    else:
        bounds = trajectory_data["_bounds"]

    lon_min, lat_min, lon_max, lat_max = bounds

    traj = trajectories[frame_idx]
    points = traj.get("points", [])
    if not points:
        return None, bounds

    count_grid = np.zeros((height, width), dtype=np.float32)

    for lon, lat in points:
        col = int((lon - lon_min) / (lon_max - lon_min) * width)
        row = int((lat_max - lat) / (lat_max - lat_min) * height)
        if 0 <= row < height and 0 <= col < width:
            count_grid[row, col] += 1.0

    if apply_smoothing:
        count_grid = gaussian_filter(count_grid, sigma=sigma)

    cache_key = f"global_max_{id(trajectory_data)}_{width}_{height}_{apply_smoothing}_{sigma}"

    if cache_key not in st.session_state:
        global_max = 0.0
        for t in trajectories:
            pts = t.get("points", [])
            if not pts:
                continue
            temp_grid = np.zeros((height, width), dtype=np.float32)
            for lon, lat in pts:
                col = int((lon - lon_min) / (lon_max - lon_min) * width)
                row = int((lat_max - lat) / (lat_max - lat_min) * height)
                if 0 <= row < height and 0 <= col < width:
                    temp_grid[row, col] += 1.0
            if apply_smoothing:
                temp_grid = gaussian_filter(temp_grid, sigma=sigma)
            global_max = max(global_max, float(temp_grid.max()))
        st.session_state[cache_key] = global_max
    else:
        global_max = st.session_state[cache_key]

    if global_max > 0:
        normalized = (count_grid / global_max) * 255.0
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    else:
        normalized = count_grid.astype(np.uint8)

    rgba = apply_colormap_rgba(normalized)
    return rgba, bounds

def create_folium_map_with_trajectory(trajectory_rgba: np.ndarray, bounds):
    """Create folium map with overlay."""
    lon_min, lat_min, lon_max, lat_max = bounds

    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    lat_diff = abs(lat_max - lat_min)
    lon_diff = abs(lon_max - lon_min)
    max_diff = max(lat_diff, lon_diff)

    if max_diff > 10:
        zoom = 6
    elif max_diff > 5:
        zoom = 8
    elif max_diff > 1:
        zoom = 10
    elif max_diff > 0.5:
        zoom = 11
    elif max_diff > 0.1:
        zoom = 12
    elif max_diff > 0.05:
        zoom = 13
    else:
        zoom = 14

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="OpenStreetMap")

    img = Image.fromarray(trajectory_rgba, "RGBA")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    img_url = f"data:image/png;base64,{img_b64}"

    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        opacity=0.8,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    folium.Rectangle(
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        color="red",
        weight=2,
        fill=False,
        opacity=0.6,
    ).add_to(m)

    folium.Marker(
        [center_lat, center_lon],
        popup="Center",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    plugins.Fullscreen().add_to(m)

    return m

st.title("🛢️ Oil Spill Trajectory Viewer")

with st.sidebar:
    st.header("📁 Upload Trajectory Files")

    uploaded_files = st.file_uploader(
        "Upload JSON trajectory files",
        type=["json"],
        accept_multiple_files=True,
        help="Select one or more trajectory JSON files"
    )

    if uploaded_files:
        if st.button("🔄 Load Trajectories", use_container_width=True):
            with st.spinner("Processing files..."):
                slicks = []
                for uploaded_file in uploaded_files:
                    metadata = extract_slick_metadata_from_upload(uploaded_file)
                    if metadata:
                        slicks.append(metadata)
                        # Store trajectory data
                        if metadata["id"] not in st.session_state.trajectory_data:
                            data = metadata["data"]
                            bounds = compute_trajectory_bounds(data)
                            if bounds is not None:
                                data["_bounds"] = bounds
                            st.session_state.trajectory_data[metadata["id"]] = data
                
                st.session_state.slicks_metadata = slicks
                st.session_state.selected_slick = None
                st.session_state.frames_cache = {}
                st.success(f"✓ {len(slicks)} trajectories loaded")

    st.divider()
    st.header("⚙️ Settings")

    apply_smoothing = st.checkbox("Smoothing", value=True)
    smoothing_sigma = st.slider("Sigma", 1.0, 10.0, 6.0, 0.5)
    fps = st.slider("FPS", 1, 30, 12)
    st.session_state.fps_setting = fps
    resolution = st.selectbox("Resolution", [500, 1000, 1500, 2000], index=1)

    st.divider()
    st.header("🎯 Slicks")

    if st.session_state.slicks_metadata:
        for slick in st.session_state.slicks_metadata:
            is_selected = (slick["id"] == st.session_state.selected_slick)
            label = f"{'🔴' if is_selected else '🔵'} {slick['id']}"
            if st.button(label, key=f"slick_{slick['id']}", use_container_width=True):
                st.session_state.playing = False
                st.session_state.current_frame = 0
                st.session_state.selected_slick = slick["id"]
                st.rerun()
    else:
        st.info("👆 Upload trajectory files")

col1, col2 = st.columns([3, 1])

with col1:
    if st.session_state.selected_slick:
        trajectory_data = st.session_state.trajectory_data.get(st.session_state.selected_slick)
        if trajectory_data:
            frame_key = (
                f"{st.session_state.selected_slick}_"
                f"{st.session_state.current_frame}_"
                f"{resolution}_"
                f"{apply_smoothing}_"
                f"{smoothing_sigma}"
            )

            if frame_key not in st.session_state.frames_cache:
                frame_data, bounds = generate_particle_count_frame(
                    trajectory_data,
                    st.session_state.current_frame,
                    resolution,
                    resolution,
                    apply_smoothing,
                    smoothing_sigma,
                )
                if frame_data is not None:
                    st.session_state.frames_cache[frame_key] = (frame_data, bounds)

            if frame_key in st.session_state.frames_cache:
                frame_data, bounds = st.session_state.frames_cache[frame_key]
                try:
                    m = create_folium_map_with_trajectory(frame_data, bounds)
                    map_html = m._repr_html_()
                    components.html(map_html, width=800, height=600, scrolling=False)
                except Exception as e:
                    st.error(f"Map error: {e}")
                    st.image(frame_data, use_column_width=True)
            else:
                st.warning("No frame data")
        else:
            st.error("Failed to load trajectory")
    else:
        st.info("👈 Select a trajectory")

with col2:
    st.header("🎬 Controls")

    if st.session_state.selected_slick:
        selected_data = next(
            (s for s in st.session_state.slicks_metadata if s["id"] == st.session_state.selected_slick),
            None,
        )

        if selected_data:
            num_frames = selected_data["num_frames"]

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("⏮️"):
                    st.session_state.playing = False
                    st.session_state.current_frame = (st.session_state.current_frame - 1) % num_frames
                    st.rerun()

            with c2:
                play_checkbox = st.checkbox(
                    "▶️ Play",
                    value=st.session_state.playing,
                    key="play_checkbox",
                )
                if play_checkbox != st.session_state.playing:
                    st.session_state.playing = play_checkbox
                    st.rerun()

                if st.session_state.playing:
                    st.success("Playing")
                else:
                    st.info("Paused")

            with c3:
                if st.button("⏭️"):
                    st.session_state.playing = False
                    st.session_state.current_frame = (st.session_state.current_frame + 1) % num_frames
                    st.rerun()

            trajectory_data = st.session_state.trajectory_data.get(st.session_state.selected_slick)
            if trajectory_data and "trajectory" in trajectory_data:
                dt = trajectory_data["trajectory"][st.session_state.current_frame].get("datetime", "N/A")
                st.info(f"Frame {st.session_state.current_frame + 1}/{num_frames}\n\n{dt}")

            if st.session_state.playing:
                st.slider("Frame", 0, num_frames - 1, st.session_state.current_frame, key="frame_slider", disabled=True)
            else:
                new_frame = st.slider("Frame", 0, num_frames - 1, st.session_state.current_frame, key="frame_slider")
                if new_frame != st.session_state.current_frame:
                    st.session_state.current_frame = new_frame
                    st.rerun()

            st.divider()

            if st.button("📥 Generate PNG"):
                frame_key = (
                    f"{st.session_state.selected_slick}_"
                    f"{st.session_state.current_frame}_"
                    f"{resolution}_"
                    f"{apply_smoothing}_"
                    f"{smoothing_sigma}"
                )
                if frame_key in st.session_state.frames_cache:
                    frame_data, _ = st.session_state.frames_cache[frame_key]
                    img = Image.fromarray(frame_data, "RGBA")
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    st.session_state.png_bytes = buf.getvalue()
                    st.success("PNG generated")

            if st.session_state.png_bytes is not None:
                st.download_button(
                    "💾 Save PNG",
                    st.session_state.png_bytes,
                    f"frame_{st.session_state.current_frame:03d}.png",
                    "image/png",
                )
    else:
        st.info("👈 Select a trajectory")

# ANIMATION LOOP
if st.session_state.playing and st.session_state.selected_slick:
    selected_data = next(
        (s for s in st.session_state.slicks_metadata if s["id"] == st.session_state.selected_slick),
        None,
    )
    if selected_data:
        num_frames = selected_data["num_frames"]
        st.session_state.current_frame = (st.session_state.current_frame + 1) % num_frames
        time.sleep(1.0 / st.session_state.fps_setting)
        st.rerun()
