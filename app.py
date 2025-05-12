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


def upload_and_store_faces(video_file, user_id, progress=gr.Progress()):
    progress(0, desc="Uploading and storing frames...")

    try:
        files = {"video": (os.path.basename(video_file.name), open(video_file.name, "rb"))}
        data = {"user_id": user_id}
        response = requests.post(f"{BASE_URL}/store_faces", files=files, data=data)

        if response.status_code != 200:
            return f"❌ Error: {response.json().get('error', 'Unknown error')}"
        
        progress(1)
        return f"✅ Successfully stored frames for user '{user_id}'"
    except Exception as e:
        return f"❌ Exception: {str(e)}"

def match_faces_by_user_id(user_id, progress=gr.Progress()):
    progress(0, desc="Matching faces...")

    try:
        response = requests.get(f"{BASE_URL}/match_faces/{user_id}")
        result = response.json()

        if response.status_code != 200:
            return [[f"❌ Error: {result.get('error', 'Unknown error')}"]], ""

        matches = result.get("matches", [])
        time_data = result.get("time", {})

        formatted_matches = [[identity] for identity in matches]
        time_str =(f"Time: {time_data}s\n")

        progress(1)
        return formatted_matches, time_str
    except Exception as e:
        return [[f"❌ Exception: {str(e)}"]], ""


# --- GRADIO UI ---

with gr.Blocks() as demo:
    gr.Markdown("# 🎯 Face Identification Interface")

    with gr.Tab("📸 Add to Database"):
        with gr.Row():
            image_input = gr.File(file_types=["image"], file_count="multiple", label="Select images")
        add_button = gr.Button("Add to Database")
        add_output = gr.JSON(label="Upload Response")

        add_button.click(fn=call_update_db, inputs=[image_input], outputs=[add_output])

    with gr.Tab("📥 Upload & Store Faces"):
        with gr.Row():
            store_video_input = gr.File(file_types=[".mp4"], file_count="single", label="Upload Video")
            store_user_id_input = gr.Textbox(label="User ID", placeholder="Enter user ID")

        store_button = gr.Button("Store Faces")
        store_status = gr.Textbox(label="Status", interactive=False)

        store_button.click(
            fn=upload_and_store_faces,
            inputs=[store_video_input, store_user_id_input],
            outputs=[store_status]
        )

    with gr.Tab("🔍 Match Faces"):
        with gr.Row():
            match_user_id_input = gr.Textbox(label="User ID", placeholder="Enter user ID")

        match_button = gr.Button("Find Matches")

        with gr.Row():
            matches_output = gr.List(
                label="Matched Identities",
                headers=["Identity"],
                interactive=False
            )
            time_output = gr.Textbox(
                label="Processing Time",
                interactive=False
            )

        match_button.click(
            fn=match_faces_by_user_id,
            inputs=[match_user_id_input],
            outputs=[matches_output, time_output]
        )

        
demo.launch(server_name="0.0.0.0", server_port=7860)
