import os
import yaml
import json
import requests
import gradio as gr

# Load config from YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BASE_URL = f"http://localhost:{config['port']}"

def initialize_api(params):
    try:
        response = requests.post(f"{BASE_URL}/initialize", json=params)

        if response.headers.get("Content-Type", "").startswith("application/json"):
            return response.json()
        else:
            return {
                "error": "Invalid response from server",
                "status_code": response.status_code,
                "text": response.text
            }
    except Exception as e:
        return {"error": str(e)}


config_json = {
    "face_detector_path": config["face_detector_path"],
    "mongo_ip": config["mongo_ip"],
    "mongo_port": config["mongo_port"],
    "every_nth": config["every_nth"],
    "max_frames": config["max_frames"],
    "model_name": config["model_name"],
    "distance_metric": config["distance_metric"],
    "enforce_detection": config["enforce_detection"],
    "detector_backend": config["detector_backend"],
    "threshold": config["threshold"],
    "device": config["device"],
    "distance_threshold": config["distance_threshold"]
}

init_response = initialize_api(config_json)
print(init_response)

# --- API CALL HELPERS ---

def call_update_db(files, progress=gr.Progress()):
    progress(0, desc="Uploading images...")

    files_data = [("images", (os.path.basename(f.name), open(f.name, "rb"))) for f in files]

    try:
        response = requests.post(f"{BASE_URL}/update_db", files=files_data)
        progress(1)
        return response.json()
    except Exception as e:
        return [f"Error: {str(e)}"]


def call_get_identities():
    try:
        response = requests.get(f"{BASE_URL}/get_identities")
        result = response.json()
        return [[identity] for identity in result.get("identities", [])]
    except Exception as e:
        return [f"Error: {str(e)}"]


def call_face_id(video_file, progress=gr.Progress()):
    progress(0, desc="Uploading video...")

    try:
        files = {"video": (os.path.basename(video_file.name), open(video_file.name, "rb"))}
        response = requests.post(f"{BASE_URL}/face_id", files=files)
        result = response.json()

        progress(1)

        matches_raw = result.get("matches", "[]")
        time_data = result.get("time", {})

        matches = json.loads(matches_raw) if isinstance(matches_raw, str) else matches_raw
        formatted_matches = [[identity] for identity in matches]
        
        time_str = (
            f"Video Processing: {time_data.get('video_processing_time', 0):.2f}s\n"
            f"Face Matching: {time_data.get('face_matching_time', 0):.2f}s\n"
            f"Total: {sum(time_data.values()):.2f}s"
        )

        progress(1)
        return formatted_matches, time_str
    except Exception as e:
        return [f"Error: {str(e)}"]


# --- GRADIO UI ---

with gr.Blocks() as demo:
    gr.Markdown("# 🎯 Face Identification Interface")

    with gr.Tab("📸 Add to Database"):
        with gr.Row():
            image_input = gr.File(file_types=["image"], file_count="multiple", label="Select images")
        add_button = gr.Button("Add to Database")
        add_output = gr.JSON(label="Upload Response")

        add_button.click(fn=call_update_db, inputs=[image_input], outputs=[add_output])

    with gr.Tab("🧠 View Identities"):
        get_button = gr.Button("Get Identities")
        identities_output = gr.List(label="Current Identities")

        get_button.click(fn=call_get_identities, inputs=[], outputs=[identities_output])

    with gr.Tab("🎥 Identify from Video"):
        with gr.Row():
            video_input = gr.File(file_types=[".mp4"], file_count="single", label="Select a video")
        
        identify_button = gr.Button("Identify Faces")

        with gr.Row():
            matches_output = gr.List(label="Matched Identities",
                                     headers=["Identity"],
                                     interactive=False)
            time_output = gr.Textbox(label="Processing Time",
                                     interactive=False)
        
        identify_button.click(fn=call_face_id,
                              inputs=[video_input],
                              outputs=[matches_output, time_output])
        
demo.launch(server_name="0.0.0.0", server_port=7860)
