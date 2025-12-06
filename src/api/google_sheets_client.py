import gspread
from google.oauth2 import service_account
from google.auth import default as google_auth_default
import os
import datetime
from typing import List, Dict, Any


class GoogleSheetsClient:
    """
    Client for interacting with Google Sheets to track video status.
    Sheet Name: "Video Log"
    Columns: Timestamp, VideoID, Prompt, Status, CombinationKey, Reward

    인증 방식:
    1. credentials_path가 존재하면 Service Account JSON 사용
    2. 없으면 Application Default Credentials 사용 (Cloud Run Workload Identity)
    """

    def __init__(self, credentials_path: str = None, sheet_id: str = None):
        self.scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive',
            'https://www.googleapis.com/auth/spreadsheets'
        ]

        try:
            # 1. credentials_path가 있고 파일이 존재하면 Service Account JSON 사용
            if credentials_path and os.path.exists(credentials_path):
                print(f"[Sheets] Using Service Account JSON: {credentials_path}")
                creds = service_account.Credentials.from_service_account_file(
                    credentials_path, scopes=self.scope
                )
                self.client = gspread.authorize(creds)
            else:
                # 2. Application Default Credentials 사용 (Cloud Run)
                print("[Sheets] Using Application Default Credentials")
                creds, project = google_auth_default(scopes=self.scope)
                self.client = gspread.authorize(creds)

            if sheet_id:
                self.sheet = self.client.open_by_key(sheet_id).sheet1
                print(f"[Sheets] Connected to sheet: {sheet_id}")
            else:
                print("[Sheets] Warning: Spreadsheet ID not provided")
                self.sheet = None

        except Exception as e:
            print(f"[Sheets] Error initializing: {e}")
            self.client = None
            self.sheet = None

    def log_video(self, video_id: str, prompt: str, combination_key: str):
        """
        Log a newly generated video.
        Columns: Timestamp, VideoID, Prompt, Status, CombinationKey, Reward
        """
        if not self.sheet: return

        timestamp = datetime.datetime.now().isoformat()
        row = [timestamp, video_id, prompt, "PENDING", combination_key, ""]
        try:
            self.sheet.append_row(row)
            print(f"[Sheets] Logged video {video_id}")
        except Exception as e:
            print(f"[Sheets] Error logging video: {e}")

    def get_pending_videos(self) -> List[Dict[str, Any]]:
        """
        Get videos that are PENDING and older than 6 hours.
        """
        if not self.sheet: return []

        try:
            records = self.sheet.get_all_records()
            pending = []
            now = datetime.datetime.now()

            for i, record in enumerate(records):
                # Check status
                if record.get("Status") != "PENDING":
                    continue

                # Check time
                ts_str = record.get("Timestamp")
                try:
                    ts = datetime.datetime.fromisoformat(ts_str)
                    age = now - ts
                    if age.total_seconds() >= 6 * 3600:  # 6 hours
                        # Return record with row index (1-based, +1 for header)
                        record['row_index'] = i + 2
                        pending.append(record)
                except ValueError:
                    continue

            return pending
        except Exception as e:
            print(f"[Sheets] Error fetching pending videos: {e}")
            return []

    def update_video_status(self, row_index: int, status: str, reward: float = None):
        """
        Update status and reward for a video.
        """
        if not self.sheet: return

        try:
            # Update Status (Column 4)
            self.sheet.update_cell(row_index, 4, status)

            # Update Reward (Column 6) if provided
            if reward is not None:
                self.sheet.update_cell(row_index, 6, reward)

            print(f"[Sheets] Updated row {row_index} to {status}")
        except Exception as e:
            print(f"[Sheets] Error updating status: {e}")