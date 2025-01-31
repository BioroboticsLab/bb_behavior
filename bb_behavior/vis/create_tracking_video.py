import numpy as np
import pandas as pd
import os
import shutil
import subprocess
import tempfile
import ast
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as mpl_cm
import matplotlib.font_manager as font_manager
from matplotlib import font_manager

from bb_utils.ids import BeesbookID


# Function to convert stringified lists back to Python lists
def convert_to_list(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)  # Safely evaluate string as Python literal
        except (ValueError, SyntaxError):
            return value  # Return as-is if conversion fails
    return value  # Return as-is if already a list

def convert_12bit_beeID_to_int(beeID_12bits):
    beeID_12bits = np.array(convert_to_list(beeID_12bits))  # csv format can mess this up, this is a safe conversion handling
    integer_id = BeesbookID.from_bb_binary((beeID_12bits.astype(float)>0.5).astype(int)).as_ferwar()
    return integer_id

# Font path for labeling
FONT_PATH = font_manager.findfont("Sans", fallback_to_default=True)

def create_tracking_video(
    video_path,
    output_video,
    video_start_timestamp,
    fps=6,
    tracks_df=None,            # DataFrame for tagged bees with track info
    video_dataframe=None,      # DataFrame for raw detections (untagged or all)
    track_history=0,           # How many past frames to connect for track tails
    draw_labels=True,
    color_mod=20,
    scale_factor=0.25,         # Scale the output video e.g. 25%
    bee_id_conf_threshold=0.8,
    detect_conf_threshold=0.8,
    font_size = 60,
    r_untagged = 20,
    r_tagged = 20
):
    """
    Creates an annotated video by combining:
      - Tagged bee trajectories (tracks_df) 
      - Untagged detections (video_dataframe)
    
    If both DataFrames are given:
      - Use `tracks_df` exclusively for tagged bees
      - Use `video_dataframe` only for untagged detections

    Steps:
      1) Combine relevant data into a single DataFrame with standard columns:
            ['bee_id', 'bee_id_confidence', 'detection_confidence',
             'timestamp', 'x_pixels', 'y_pixels', 'detection_type']
         Threshold values the given confidences.
      2) Convert timestamps → frame_number (1-based) based on `video_start_timestamp` + `fps`.
      3) Extract frames from the source video to a temporary folder.
      4) Annotate frames with lines, labels, track tails, etc.
      5) Re-encode annotated frames → output video.
      6) Cleanup the temporary folder.

    Parameters
    ----------
    video_path : str
        Path to the source MP4 video.
    output_video : str
        Output path for the annotated MP4.
    video_start_timestamp : str or pd.Timestamp
        Start time of the video for mapping timestamps → frames.
    fps : float
        Frame rate to assume for time→frame conversion (1 frame = 1/fps seconds).
    tracks_df : pd.DataFrame or None
        DataFrame for **tagged** bees with columns like:
          ['bee_id', 'bee_id_confidence', 'track_id', 'detection_confidence',
           'x_pixels', 'y_pixels', 'timestamp', 'detection_type', ...]
        If None, no tagged trajectories are drawn.
    video_dataframe : pd.DataFrame or None
        DataFrame for raw detections with columns:
          ['localizerSaliency', 'beeID', 'xpos', 'ypos', 'confidence',
           'timestamp', 'detection_type', ...]
        If None, no untagged detections are drawn.
    track_history : int
        Number of **past frames** to connect for track tails (only applies to rows with track_id).
    draw_labels : bool
        Whether to draw text labels (bee_id) near the first point of each track in a frame.
    color_mod : int
        For stable color assignment, color = colormap(bee_id % color_mod).
    scale_factor : float
        Scale factor for final encoded video (e.g. 0.25 = quarter size).
    bee_id_conf_threshold : float
    detect_conf_threshold : float

    Returns
    -------
    None. Writes an annotated video file at `output_video`.
    """

    # 0) Build a final DataFrame with standardized columns:
    #    bee_id, bee_id_confidence, track_id, detection_confidence,
    #    timestamp, x_pixels, y_pixels, detection_type, frame_number
    # We'll do so by combining tracks_df (tagged) and video_dataframe (untagged).

    frames_list = []

    ### A) Handle the tracks_df (tagged bees)
    if tracks_df is not None and not tracks_df.empty:
        df_tagged = tracks_df.copy()
        # filter by confiendence
        if "bee_id_confidence" in df_tagged.columns:
            df_tagged = df_tagged[df_tagged["bee_id_confidence"]> bee_id_conf_threshold]

        if "detection_confidence" in df_tagged.columns:
            df_tagged = df_tagged[df_tagged["detection_confidence"]> detect_conf_threshold]

        # Ensure columns exist for x_pixels,y_pixels,detection_type,track_id
        for col in ["track_id", "bee_id", "detection_type"]:
            if col not in df_tagged.columns:
                df_tagged[col] = np.nan

        # We'll keep only tagged bees from this DataFrame, i.e. detection_type == 1 typically
        # If there's no consistent definition, you can skip this step or filter as needed.
        # For safety, let's just keep them all, the user mentioned "If both are passed in => only use trajectories of tagged bees here"
        # So let's forcibly set detection_type=1 if not set:
        if "detection_type" in df_tagged.columns:
            # If your data has 1 for tagged, you can filter or set them:
            # df_tagged = df_tagged[df_tagged["detection_type"] == 1]
            df_tagged["detection_type"] = df_tagged["detection_type"].fillna(1)
        else:
            df_tagged["detection_type"] = 1

        # Standardize columns
        df_tagged = df_tagged.rename(columns={
            # we expect them to exist, but just to be sure:
            "x_pixels": "x_pixels",
            "y_pixels": "y_pixels",
        })

        # Keep only relevant columns
        keep_cols = [
            "bee_id",
            "bee_id_confidence",
            "track_id",
            "detection_confidence",
            "timestamp",
            "x_pixels",
            "y_pixels",
            "detection_type"
        ]
        df_tagged = df_tagged[keep_cols].copy()

        frames_list.append(df_tagged)

    ### B) Handle the video_dataframe (untagged or raw detections)
    if video_dataframe is not None and not video_dataframe.empty:
        df_un = video_dataframe.copy()
        # check if its the rpi style timestamps, which I saved as float from the start of the video
        if np.min(df_un['timestamp'])<10000:
            df_un['timestamp'] = pd.to_timedelta(df_un['timestamp'], unit='s') + video_start_timestamp
        
        # make sure its a datetime type
        df_un['timestamp'] = pd.to_datetime(df_un['timestamp'],unit='s',utc=True)

        # Convert 12-bit beeID => float
        if "beeID" in df_un.columns:
            sel = df_un['detection_type']=='TaggedBee'
            df_un.loc[sel,"bee_id"] = df_un.loc[sel,"beeID"].apply(convert_12bit_beeID_to_int)
        else:
            # fallback
            df_un["bee_id"] = np.nan

        # 'confidence' => bee_id_confidence
        if "confidence" in df_un.columns:
            df_un = df_un[np.logical_not((df_un["confidence"]< bee_id_conf_threshold) & (df_un["detection_type"]=='TaggedBee'))]

        # 'localizerSaliency' => detection_confidence
        if "localizerSaliency" in df_un.columns:
            df_un = df_un[df_un["localizerSaliency"]> detect_conf_threshold]

        # 'xpos' => x_pixels, 'ypos' => y_pixels, detection_type might be present
        df_un.rename(columns={
            "xpos": "x_pixels",
            "ypos": "y_pixels"
        }, inplace=True)


        # Keep relevant columns
        keep_cols = [
            "bee_id",
            "bee_id_confidence",
            "detection_confidence",
            "timestamp",
            "x_pixels",
            "y_pixels",
            "detection_type"  # if missing, it will become NaN
        ]
        # Guarantee the columns exist
        for c in keep_cols:
            if c not in df_un.columns:
                df_un[c] = np.nan

        df_un = df_un[keep_cols].copy()

        # If we ALSO have tracks_df, user says "only use trajectories of tagged bees from tracks_df"
        # => That means in df_un we keep only untagged detection_type != 1 
        #    or if detection_type is missing, we treat it as untagged
        if tracks_df is not None:
            # Filter out any "tagged" from df_un
            df_un = df_un[(df_un["detection_type"] != 'TaggedBee') & (df_un["detection_type"].notna())]

        frames_list.append(df_un)

    # If both are None, there's nothing to annotate
    if not frames_list:
        print("No data to show (tracks_df and video_dataframe are both None or empty).")
        return

    # Combine into one final DataFrame
    df_final = pd.concat(frames_list, ignore_index=True)
    df_final.dropna(subset=["timestamp"], inplace=True)  # ensure we have timestamps
    df_final.sort_values("timestamp", inplace=True)

    # 1) Convert timestamps => frame_number
    if not pd.api.types.is_datetime64_any_dtype(df_final["timestamp"]):
        df_final["timestamp"] = pd.to_datetime(df_final["timestamp"],format="mixed")
    video_start_ts = pd.Timestamp(video_start_timestamp)
    df_final["relative_secs"] = (df_final["timestamp"] - video_start_ts).dt.total_seconds()
    # Using +1 so frames start at 1 (as your code does for ffmpeg extraction)
    df_final["frame_number"] = (df_final["relative_secs"] * fps).round().astype(int) + 1

    # Decide columns for X/Y
    x_col = "x_pixels"
    y_col = "y_pixels"

    # 2) Extract frames from the video => random temp folder
    tmp_dir = tempfile.mkdtemp(prefix="tracking_frames_")
    try:
        print(f"[INFO] Extracting frames from {video_path} into {tmp_dir} ...")
        extract_cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vsync", "0",
            "-qscale:v", "2",
            os.path.join(tmp_dir, "frame_%06d.jpg")
        ]
        subprocess.run(extract_cmd, check=True)
        print("[INFO] Extraction complete.")

        # 3) Annotate frames
        print("[INFO] Annotating frames...")

        # List extracted frames
        frame_files = sorted(
            f for f in os.listdir(tmp_dir)
            if f.startswith("frame_") and f.endswith(".jpg")
        )

        def annotate_frame(img_path, frame_idx):
            """
            Annotate a single frame in-place by drawing track tails for tagged bees (bee_id is not NaN)
            or an orange ellipse for untagged bees (bee_id is NaN), skipping any track history for untagged.
            """
            # We'll gather data in [frame_idx - track_history, frame_idx] for tagged bees only.
            # For untagged, we skip tail logic altogether.
            min_f = max(frame_idx - track_history, 1)
        
            data_sub = df_final[
                (df_final["frame_number"] >= min_f) & (df_final["frame_number"] <= frame_idx)
            ].copy()
        
            if data_sub.empty:
                return
        
            # Load image
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img, "RGBA")
        
            # Prepare font
            font = ImageFont.truetype(FONT_PATH, font_size)

            # 1) Annotate Untagged bees
            df_g = data_sub[(data_sub['bee_id'].isna())]
            # UNTAGGED BEE(S):
            #  - No track history
            #  - Just draw an orange circle for the current frame only
            #  - (Optional) no labels
            df_current = df_g[df_g["frame_number"] == frame_idx]

            # Draw an orange ellipse for each detection
            for _, row_c in df_current.iterrows():
                cx = row_c[x_col]
                cy = row_c[y_col]
                # RGBA => (255,165,0) is orange in RGB; alpha=150 for partial fill
                fill_color = (255, 165, 0, 150)
                outline_color = (255, 165, 0, 255)
                draw.ellipse(
                    [(cx - r_untagged, cy - r_untagged), (cx + r_untagged, cy + r_untagged)],
                    fill=fill_color,
                    outline=outline_color
                )        
            
            # 2) Group by 'bee_id': tagged bees
            for bee_id_val, df_g in data_sub.groupby("bee_id"):
                df_g = df_g.sort_values("frame_number")
                if df_g.empty:
                    continue
                
                # TAGGED BEE (bee_id is numeric):
                # => we draw tails for the last 'track_history' frames plus the current frame
                try:
                    numeric_id = int(bee_id_val)
                except:
                    numeric_id = -1
        
                fraction = (numeric_id % color_mod) / float(color_mod)
                (rc, gc, bc, ac) = mpl_cm.hsv(fraction)  # 0..1
                r255, g255, b255 = int(rc * 255), int(gc * 255), int(bc * 255)
        
                # partial alpha for fill, full alpha for lines
                color_fill = (r255, g255, b255, 100)
                color_line = (r255, g255, b255, 255)
        
                # Gather points for potential tail
                points = []
                for _, row_i in df_g.iterrows():
                    fn = row_i["frame_number"]
                    px = row_i[x_col]
                    py = row_i[y_col]
                    points.append((fn, px, py))
        
                points.sort(key=lambda x: x[0])
        
                # Draw lines across frames for tail
                for i in range(1, len(points)):
                    fn0, x0, y0 = points[i - 1]
                    fn1, x1, y1 = points[i]
                    # Only connect if frames are within track_history distance
                    if fn1 - fn0 <= track_history:
                        draw.line([(x0, y0), (x1, y1)], fill=color_line, width=3)
        
                # Now draw dots for the current frame only
                df_current = df_g[df_g["frame_number"] == frame_idx]
                if df_current.empty:
                    continue
        
                last_xy = None
                first_label = True
        
                # Sort for consistent draw order
                df_current = df_current.sort_values(x_col)
        
                for _, row_c in df_current.iterrows():
                    cx = row_c[x_col]
                    cy = row_c[y_col]
                    r = r_tagged
                    draw.ellipse(
                        [(cx - r, cy - r), (cx + r, cy + r)],
                        fill=color_fill,
                        outline=color_line
                    )
        
                    last_xy = (cx, cy)
        
                    # Draw label once
                    if draw_labels and first_label:
                        label_str = f"{numeric_id}"
                        draw.text((cx + 1.5 * r, cy - 2 * r),
                                  label_str, fill=color_line, font=font)
                        first_label = False
        
            # Save annotated
            img.save(img_path)
            img.close()

        # go frame by frame
        for fpath in frame_files:
            # e.g. frame_000123.jpg => 123
            parts = fpath.split("_")
            n_str = parts[1].split(".")[0]
            frame_idx = int(n_str)
            full_path = os.path.join(tmp_dir, fpath)
            annotate_frame(full_path, frame_idx)

        print("[INFO] Annotation complete.")

        # 4) Encode to final video
        if os.path.exists(output_video):
            os.remove(output_video)

        print(f"[INFO] Re-encoding to {output_video} ...")
        scale_expr = f"scale=trunc(iw*{scale_factor}/2)*2:trunc(ih*{scale_factor}/2)*2"
        encode_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%06d.jpg"),
            "-vf", scale_expr,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_video
        ]
        subprocess.run(encode_cmd, check=True)
        print("[INFO] Encoding done.")

    finally:
        # 5) Cleanup
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        print(f"[INFO] Temporary folder {tmp_dir} removed.")
        print(f"[INFO] Final annotated video: {output_video}")