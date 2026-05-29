from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool


class PredictPixelsRequest(BaseModel):
    pixels: list[float] = Field(..., min_length=784, max_length=784)

    @field_validator("pixels")
    @classmethod
    def check_pixel_range(cls, values: list[float]) -> list[float]:
        for v in values:
            if v < 0.0 or v > 255.0:
                raise ValueError("pixel values must be 0-255")
        return values


class PredictGridRequest(BaseModel):
    image: list[list[float]]

    @field_validator("image")
    @classmethod
    def check_grid_shape(cls, grid: list[list[float]]) -> list[list[float]]:
        if len(grid) != 28:
            raise ValueError("image must be 28 rows")
        for row in grid:
            if len(row) != 28:
                raise ValueError("each row must be 28 columns")
            for v in row:
                if v < 0.0 or v > 255.0:
                    raise ValueError("pixel values must be 0-255")
        return grid


class PredictResponse(BaseModel):
    digit: int = Field(..., ge=0, le=9)
    confidence: float = Field(..., ge=0.0, le=1.0)
