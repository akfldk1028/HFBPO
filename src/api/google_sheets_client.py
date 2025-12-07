import gspread
from google.oauth2 import service_account
from google.auth import default as google_auth_default
import os
import datetime
from typing import List, Dict, Any


class GoogleSheetsClient:
    """
    Google Sheets 연동 클라이언트
    컬럼: Timestamp, VideoID, Prompt, Status, CombinationKey, Reward
    """

    def __init__(self, credentials_path: str = None, sheet_id: str = None):
        self.scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive',
            'https://www.googleapis.com/auth/spreadsheets'
        ]

        try:
            if credentials_path and os.path.exists(credentials_path):
                # Service Account JSON
                creds = service_account.Credentials.from_service_account_file(
                    credentials_path, scopes=self.scope
                )
            else:
                # ADC (Cloud Run용)
                creds, _ = google_auth_default(scopes=self.scope)

            self.client = gspread.authorize(creds)
            self.sheet = self.client.open_by_key(sheet_id).sheet1 if sheet_id else None

        except Exception as e:
            print(f"[Sheets] Init error: {e}")
            self.client = None
            self.sheet = None

    def log_video(self, video_id: str, prompt: str, combination_key: str):
        """새 영상 로그 추가"""
        if not self.sheet:
            return

        row = [
            datetime.datetime.now().isoformat(),
            video_id,
            prompt,
            "PENDING",
            combination_key,
            ""
        ]
        try:
            self.sheet.append_row(row)
        except Exception as e:
            print(f"[Sheets] Log error: {e}")

    def get_pending_videos(self) -> List[Dict[str, Any]]:
        """6시간 지난 PENDING 영상 조회"""
        if not self.sheet:
            return []

        try:
            records = self.sheet.get_all_records()
            pending = []
            now = datetime.datetime.now()

            for i, record in enumerate(records):
                if record.get("Status") != "PENDING":
                    continue

                ts_str = record.get("Timestamp")
                try:
                    ts = datetime.datetime.fromisoformat(ts_str)
                    if (now - ts).total_seconds() >= 6 * 3600:
                        record['row_index'] = i + 2  # 헤더 +1, 0-index +1
                        pending.append(record)
                except:
                    continue

            return pending
        except Exception as e:
            print(f"[Sheets] Fetch error: {e}")
            return []

    def update_video_status(self, row_index: int, status: str, reward: float = None):
        """상태 및 보상 업데이트"""
        if not self.sheet:
            return

        try:
            self.sheet.update_cell(row_index, 4, status)
            if reward is not None:
                self.sheet.update_cell(row_index, 6, reward)
        except Exception as e:
            print(f"[Sheets] Update error: {e}")
