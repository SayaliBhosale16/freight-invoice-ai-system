from pydantic import BaseModel, ConfigDict, Field


# --- Freight Prediction ---

class FreightPredictionRequest(BaseModel):
    dollars: float = Field(gt=0, description="Invoice dollar amount")


class FreightBatchRequest(BaseModel):
    items: list[FreightPredictionRequest] = Field(max_length=100)


class FreightPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    dollars: float
    predicted_freight: float
    model_version: str


# --- Invoice Risk Flagging ---

class InvoiceRiskRequest(BaseModel):
    invoice_quantity: float = Field(gt=0)
    invoice_dollars: float = Field(gt=0)
    freight: float = Field(ge=0)
    days_po_to_invoice: float = Field(default=0, ge=0)
    days_to_pay: float = Field(default=0, ge=0)
    total_brands: float = Field(default=1, ge=0)
    total_item_quantity: float = Field(gt=0)
    total_item_dollars: float = Field(gt=0)
    avg_receiving_time: float = Field(default=0, ge=0)


class InvoiceRiskBatchRequest(BaseModel):
    items: list[InvoiceRiskRequest] = Field(max_length=100)


class InvoiceRiskResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    risk_flag: int
    risk_label: str
    confidence: float
    model_version: str


# --- Retraining ---

class RetrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_name: str
    old_version: str
    new_version: str
    old_metrics: dict
    new_metrics: dict
    promoted: bool


# --- Health / Info ---

class ModelInfo(BaseModel):
    version: str
    loaded: bool
    metrics: dict


class HealthResponse(BaseModel):
    status: str
    models: dict[str, ModelInfo]
    prediction_count_24h: int
