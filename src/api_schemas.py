"""
Pydantic request/response schemas for the AML Inference API.

All incoming JSON payloads are validated against these models before
reaching inference logic. Invalid payloads are rejected with HTTP 422.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    account_id: Optional[str] = None
    amount: float = Field(..., gt=0, description="Transaction amount, must be positive")
    mcc: Optional[str] = Field(None, max_length=10)
    payment_type: Optional[str] = Field(None, max_length=50)
    device_change: Optional[int] = Field(None, ge=0, le=1)
    ip_risk: Optional[float] = Field(None, ge=0.0, le=1.0)
    count_1h: Optional[int] = Field(None, ge=0)
    sum_24h: Optional[float] = Field(None, ge=0.0)
    uniq_payees_24h: Optional[int] = Field(None, ge=0)
    country: Optional[str] = Field(None, max_length=10)


class SequenceRequest(BaseModel):
    events: List[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Ordered list of event type strings",
    )


class ConsolidateRequest(BaseModel):
    account_id: Optional[str] = Field(None, max_length=100)
    transaction: TransactionRequest
    events: Optional[List[str]] = Field(default=[], max_length=500)


class BatchItem(BaseModel):
    account_id: Optional[str] = Field(None, max_length=100)
    transaction: TransactionRequest
    events: Optional[List[str]] = Field(default=[], max_length=500)


class BatchRequest(BaseModel):
    transactions: List[BatchItem] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of transaction items to score",
    )
