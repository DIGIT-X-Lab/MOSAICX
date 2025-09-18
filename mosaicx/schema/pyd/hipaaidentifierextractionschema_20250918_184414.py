from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

class HIPAAIdentifierExtractionSchema(BaseModel):
    """Schema for extracting personal health identifiers as defined by the HIPAA Privacy Rule."""
    full_name: str = Field(None, description="Full name of the individual.")
    address: Dict[str, Any] = Field(None, description="Street address and related geographic details.")
    date_of_birth: str = Field(None, description="Date of birth (year, month, day).")
    phone_numbers: List[str] = Field(None, description="List of phone numbers associated with the individual.")
    fax_numbers: List[str] = Field(None, description="List of fax numbers associated with the individual.")
    email_addresses: List[str] = Field(None, description="List of email addresses associated with the individual.")
    social_security_number: str = Field(None, description="Social Security Number (SSN) in the format XXX-XX-XXXX.")
    medical_record_number: str = Field(None, description="Medical record number assigned by a health care provider.")
    health_plan_beneficiary_number: str = Field(None, description="Health plan beneficiary number.")
    account_number: str = Field(None, description="Account number (e.g., for billing).")
    certificate_license_number: str = Field(None, description="Certificate or license number.")
    vehicle_identifier: str = Field(None, description="Vehicle identifier or serial number, including license plate numbers.")
    device_identifier: str = Field(None, description="Device identifier or serial number.")
    web_urls: List[str] = Field(None, description="Web URLs associated with the individual.")
    ip_addresses: List[str] = Field(None, description="IP address numbers associated with the individual.")
    biometric_identifier: str = Field(None, description="Biometric identifier (e.g., fingerprint, voice print).")
    facial_image_url: str = Field(None, description="URL to a full-face photographic image or comparable image.")
    other_unique_identifier: str = Field(None, description="Any other unique identifying number, characteristic, or code not covered above.")
