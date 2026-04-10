from pydantic import BaseModel, Field
from typing import Optional


class CustomerInput(BaseModel):
    """
    Request body — all features the model expects.
    All fields are optional with sensible defaults
    so you can test with partial data.
    """
    MonthlyCharge       : Optional[float] = Field(50.0,  description="Monthly bill amount")
    MonthsInService     : Optional[int]   = Field(12,    description="Tenure in months")
    TotalRecurringCharge: Optional[float] = Field(600.0, description="Total charges to date")
    DroppedCalls        : Optional[int]   = Field(0,     description="Number of dropped calls")
    ReceivedCalls       : Optional[int]   = Field(100,   description="Total received calls")
    MadeCallsCount      : Optional[int]   = Field(80,    description="Total outgoing calls")
    MinutesOfUse        : Optional[float] = Field(300.0, description="Total minutes used")
    OverageMinutes      : Optional[float] = Field(0.0,   description="Overage minutes used")
    NumberOfComplaints  : Optional[int]   = Field(0,     description="Complaints filed")
    HandsetPrice        : Optional[float] = Field(200.0, description="Price of handset")
    CurrentEquipmentDays: Optional[int]   = Field(180,   description="Days with current device")
    AdjustmentsToCreditRating: Optional[int] = Field(0,  description="Credit adjustments")
    RetentionCalls      : Optional[int]   = Field(0,     description="Retention calls made")
    RetentionOffersAccepted: Optional[int]= Field(0,     description="Offers accepted")

    class Config:
        json_schema_extra = {
            "example": {
                "MonthlyCharge"        : 75.5,
                "MonthsInService"      : 8,
                "TotalRecurringCharge" : 604.0,
                "DroppedCalls"         : 5,
                "ReceivedCalls"        : 120,
                "MadeCallsCount"       : 95,
                "MinutesOfUse"         : 450.0,
                "OverageMinutes"       : 20.0,
                "NumberOfComplaints"   : 2,
                "HandsetPrice"         : 150.0,
                "CurrentEquipmentDays" : 90,
                "AdjustmentsToCreditRating": 1,
                "RetentionCalls"       : 1,
                "RetentionOffersAccepted": 0
            }
        }


class ChurnResponse(BaseModel):
    """Response body returned by POST /score"""
    churn_probability : float
    risk_tier         : str   # "high" | "medium" | "low"
    will_churn        : bool
    threshold_used    : float
    recommendation    : str


class HealthResponse(BaseModel):
    status  : str
    model   : str
    version : str