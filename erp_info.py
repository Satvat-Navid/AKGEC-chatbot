from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

import requests


BASE_URL = "https://erp.akgec.ac.in"

logger = logging.getLogger(__name__)


# TypedDict definitions preserve the original JSON key format.
Identity = TypedDict(
    "Identity",
    {
        "Name": str,
        "Login": Any,
        "User ID": Any,
        "Admission number": Any,
        "Roll number": Any,
    },
)

Academic = TypedDict(
    "Academic",
    {
        "Batch ID": Any,
        "Branch ID": Any,
        "Admission date": Any,
        "Status": Any,
        "Academic year": Any,
    },
)

Contact = TypedDict(
    "Contact",
    {
        "Email": Any,
        "Student mobile": Any,
        "Parent mobile": Any,
    },
)

Personal = TypedDict(
    "Personal",
    {
        "Date of birth": Any,
        "Gender": Any,
        "Blood group": Any,
        "Aadhaar": str | None,
    },
)

Family = TypedDict(
    "Family",
    {
        "Father": Any,
        "Mother": Any,
        "Parent occupation": Any,
    },
)

Address = TypedDict(
    "Address",
    {
        "Address": Any,
        "Religion": Any,
    },
)

PreviousEducation = TypedDict(
    "PreviousEducation",
    {
        "10th percentage": Any,
        "12th percentage": Any,
        "10th board": Any,
    },
)

UserSummary = TypedDict(
    "UserSummary",
    {
        "Identity": Identity,
        "Academic": Academic,
        "Contact": Contact,
        "Personal": Personal,
        "Family": Family,
        "Address": Address,
        "Previous education": PreviousEducation,
    },
)

AttendanceSubject = TypedDict(
    "AttendanceSubject",
    {
        "Code": Any,
        "Subject": Any,
        "Present": Any,
        "Total": Any,
        "Absent": Any,
        "Percent": Any,
    },
)

AttendanceData = TypedDict(
    "AttendanceData",
    {
        "Overall attendance": Any,
        "Present": Any,
        "Total lectures": Any,
        "Remedial": Any,
        "Subjects": list[AttendanceSubject],
    },
)

ERPData = TypedDict(
    "ERPData",
    {
        "User summary": UserSummary,
        "Attendance": AttendanceData,
    },
)


class ERPError(Exception):
    """Raised when an ERP operation fails."""


class ERPClient:
    """Reusable client for fetching AKGEC user and attendance data."""

    def __init__(
        self,
        username: str,
        password: str,
        *,
        base_url: str = BASE_URL,
        timeout: int = 30,
    ) -> None:
        self.username = username
        self.password = password
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/153.0.0.0 Safari/537.36 Edg/153.0.0.0"
            )
        })

        self.auth: dict[str, Any] | None = None
        self.headers: dict[str, str] | None = None

    def __enter__(self) -> "ERPClient":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()

    def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> requests.Response:
        """Send a request with safe production logging."""
        request_headers = kwargs.get("headers", {})

        logger.debug("%s %s", method, url)
        logger.debug(
            "Request header names: %s",
            sorted(request_headers.keys()),
        )
        logger.debug(
            "Session cookie names: %s",
            sorted(self.session.cookies.get_dict().keys()),
        )

        try:
            response = self.session.request(
                method,
                url,
                timeout=self.timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            logger.exception("ERP network request failed")
            raise ERPError("ERP network request failed") from exc

        logger.debug("Response status: %s", response.status_code)
        return response

    def login(self) -> None:
        """Authenticate and prepare headers for API requests."""
        url = f"{self.base_url}/Token"

        payload = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
        }

        headers = {
            "Accept": "*/*",
            "Accept-Language": "en-GB,en;q=0.9,en-US;q=0.8",
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": self.base_url,
            "Referer": f"{self.base_url}/",
            "X-Requested-With": "XMLHttpRequest",
        }

        response = self._request(
            "POST",
            url,
            data=payload,
            headers=headers,
        )

        if response.status_code != 200:
            logger.error(
                "Login failed: status=%s response=%s",
                response.status_code,
                response.text[:500],
            )
            raise ERPError(
                f"ERP login failed with status {response.status_code}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ERPError("Login returned invalid JSON") from exc

        required_fields = [
            "access_token",
            "SessionId",
            "X-UserId",
            "X_Token",
        ]

        missing_fields = [
            field for field in required_fields if not data.get(field)
        ]

        if missing_fields:
            logger.error("Login response is missing required fields")
            raise ERPError(
                f"Login response missing fields: {missing_fields}"
            )

        self.auth = {
            "access_token": data["access_token"],
            "SessionId": data["SessionId"],
            "X_UserId": data["X-UserId"],
            "X_Token": data["X_Token"],
        }

        self.headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": (
                "en-GB,en;q=0.9,en-US;q=0.8,"
                "hi;q=0.7,hi-IN;q=0.6"
            ),
            "Authorization": f"Bearer {data['access_token']}",
            "Referer": (
                f"{self.base_url}/attendance/"
                "stu_atdance_cal_college"
            ),
            "X-Wb": "1",
            "Sessionid": str(data["SessionId"]),
            "X-Contextid": "194",
            "X-Userid": str(data["X-UserId"]),
            "X_Token": str(data["X_Token"]),
            "X-Rx": "1",
            "X_App_Year": "2026",
        }

        logger.info("ERP login successful")

    def _get_json(
        self,
        url: str,
        *,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform an authenticated GET request."""
        if self.headers is None:
            raise ERPError("Client is not logged in")

        response = self._request(
            "GET",
            url,
            params=params,
            headers=self.headers,
        )

        if response.status_code in (401, 403):
            raise ERPError("ERP authentication expired or unauthorized")

        if response.status_code != 200:
            logger.error(
                "GET request failed: status=%s response=%s",
                response.status_code,
                response.text[:500],
            )
            raise ERPError(
                f"ERP request failed with status {response.status_code}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ERPError("ERP returned invalid JSON") from exc

        if not isinstance(data, dict):
            raise ERPError("ERP returned an unexpected JSON structure")

        return data

    def get_user_details(self) -> dict[str, Any]:
        """Fetch raw user details from the ERP."""
        if self.auth is None:
            raise ERPError("Client is not logged in")

        user_id = self.auth["X_UserId"]

        return self._get_json(
            f"{self.base_url}/api/User",
            params={
                "Id": user_id,
                "val": 0,
                "val1": 0,
                "val2": 0,
                "val3": 0,
            },
        )

    def get_attendance(self) -> dict[str, Any]:
        """Fetch raw attendance data from the ERP."""
        if self.auth is None:
            raise ERPError("Client is not logged in")

        user_id = self.auth["X_UserId"]

        return self._get_json(
            f"{self.base_url}/api/SubjectAttendance/"
            "GetPresentAbsentStudent",
            params={
                "isDateWise": "false",
                "termId": 0,
                "userId": user_id,
                "y": 0,
            },
        )

    @staticmethod
    def _parse_user_details(value: Any) -> dict[str, Any]:
        """Convert userDetails JSON text into a dictionary."""
        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                logger.warning("Invalid userDetails JSON received")

        return {}

    @staticmethod
    def _make_user_summary(user: dict[str, Any]) -> UserSummary:
        """Convert raw user data into the public user-summary format."""
        first_name = user.get("firstName") or ""
        middle_name = user.get("middleName") or ""
        last_name = user.get("lastName") or ""

        full_name = " ".join(
            part for part in (first_name, middle_name, last_name) if part
        )

        aadhaar = str(user.get("aadhaarNumber") or "")
        masked_aadhaar = (
            f"{'*' * 8}{aadhaar[-4:]}"
            if len(aadhaar) >= 4
            else None
        )

        details = ERPClient._parse_user_details(
            user.get("userDetails")
        )

        return {
            "Identity": {
                "Name": full_name,
                "Login": user.get("loginName"),
                "User ID": user.get("userId"),
                "Admission number": user.get("admissionNumber"),
                "Roll number": user.get("rollNumber"),
            },
            "Academic": {
                "Batch ID": user.get("batchId"),
                "Branch ID": user.get("branchId"),
                "Admission date": user.get("admissionDate"),
                "Status": user.get("activeSubStatus"),
                "Academic year": user.get("financialYear"),
            },
            "Contact": {
                "Email": user.get("email"),
                "Student mobile": user.get("smsMobileNumber"),
                "Parent mobile": user.get("parentMobileNumber"),
            },
            "Personal": {
                "Date of birth": user.get("dob"),
                "Gender": user.get("gender"),
                "Blood group": user.get("bloodGroup"),
                "Aadhaar": masked_aadhaar,
            },
            "Family": {
                "Father": user.get("fatherName"),
                "Mother": user.get("motherName"),
                "Parent occupation": user.get("parentOccupation"),
            },
            "Address": {
                "Address": user.get("address"),
                "Religion": user.get("religion"),
            },
            "Previous education": {
                "10th percentage": details.get("tenthClassPercentage"),
                "12th percentage": (
                    details.get("twelfthClassPercentage")
                    or details.get("twelthClassPercentage")
                ),
                "10th board": (
                    details.get("tenthBoardName")
                    or details.get("tenthboardName")
                ),
            },
        }

    @staticmethod
    def _make_attendance_summary(
        attendance: dict[str, Any],
    ) -> AttendanceData:
        """Convert raw attendance into the public attendance format."""
        summary = attendance.get("stdSubAtdDetails") or {}
        raw_subjects = summary.get("subjects") or []

        subjects: list[AttendanceSubject] = []

        for subject in raw_subjects:
            if not isinstance(subject, dict):
                continue

            subjects.append(
                {
                    "Code": subject.get("code"),
                    "Subject": subject.get("name"),
                    "Present": subject.get("presentLeactures", 0),
                    "Total": subject.get("totalLeactures", 0),
                    "Absent": subject.get("absentLeactures", 0),
                    "Percent": subject.get(
                        "percentageAttendance",
                        0,
                    ),
                }
            )

        return {
            "Overall attendance": summary.get("overallPercentage"),
            "Present": summary.get("overallPresent"),
            "Total lectures": summary.get("overallLecture"),
            "Remedial": {
                "Present": summary.get("overallRemidialClass", 0),
                "Total": summary.get("overallRemidialTotalClass", 0),
            },
            "Subjects": subjects,
        }

    def fetch_data(self) -> ERPData:
        """Login once and return structured user and attendance data."""
        self.login()

        try:
            user_details = self.get_user_details()
        except ERPError as exc:
            raise ERPError("Failed to fetch user details") from exc

        try:
            attendance = self.get_attendance()
        except ERPError as exc:
            raise ERPError("Failed to fetch attendance") from exc

        return {
            "User summary": self._make_user_summary(user_details),
            "Attendance": self._make_attendance_summary(attendance),
        }