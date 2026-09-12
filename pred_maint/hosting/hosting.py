from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="pred_maint/deployment",     # the local folder containing my files
    repo_id = "mrhea/Predictive-Maintenance-Capstone"              # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
