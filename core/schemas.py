"""Typed request/response schemas for the property-valuation pipeline."""

from typing import Annotated

from pydantic import BaseModel, TypeAdapter, model_validator


class PipelineInput(BaseModel):
    """
    Structured feature payload accepted by the prediction endpoint.

    The schema holds both numerical and categorical attributes. Fields tagged
    with the ``"categorical"`` annotation are automatically detected by helper
    methods so that the preprocessing pipeline can apply the appropriate
    encoders.

    Attributes:
        type: Building type (categorical).
        sector: City sector or neighbourhood (categorical).
        net_usable_area: Net usable area in square metres.
        net_area: Net built area in square metres.
        n_rooms: Number of rooms.
        n_bathroom: Number of bathrooms.
        latitude: Geographic latitude of the property.
        longitude: Geographic longitude of the property.
    """

    type: Annotated[str, "categorical"]
    sector: Annotated[str, "categorical"]
    net_usable_area: float
    net_area: float
    n_rooms: float
    n_bathroom: float
    latitude: float
    longitude: float

    @classmethod
    def get_categorical_fields(cls) -> list[str]:
        """Return field names tagged as categorical."""
        return [
            name
            for name, field in cls.model_fields.items()
            if "categorical" in field.metadata
        ]

    @classmethod
    def get_features(cls) -> list[str]:
        """
        Return all feature names in declaration order.

        Maintaining the original order ensures that training and inference agree
        on column positions.
        """
        return list(cls.model_fields.keys())

    @classmethod
    def validate_many(cls, data: list[dict]) -> list["PipelineInput"]:
        """
        Validate a list of payload dictionaries in bulk.

        Args:
            data: A list of raw JSON-like dictionaries representing feature
                payloads.

        Returns:
            A list of fully validated :class:`PipelineInput` instances.
        """
        return TypeAdapter(list[cls]).validate_python(data)


class PipelineOutput(BaseModel):
    """
    Single-value prediction payload returned by the API.

    Attributes:
        price: Estimated market value.
    """

    price: float

    @classmethod
    @model_validator(mode="before")
    def ensure_single_field(cls, values: dict) -> dict:
        """
        Validate that exactly one field is declared at class definition.

        Although the method runs at *instance* validation time, the check looks
        at :pyattr:`cls.model_fields`, i.e. the class-level schema, to guarantee
        the output model remains a one-dimensional structure.

        Raises:
            ValueError: If more than one field is defined.
        """
        if len(cls.model_fields) != 1:
            raise ValueError("Output model must have exactly one field.")
        return values

    @classmethod
    def get_target(cls) -> str:
        """Return the sole target field name (e.g. ``'price'``)."""
        return next(iter(cls.model_fields))

    @classmethod
    def from_prediction(cls, value: float) -> "PipelineOutput":
        """
        Instantiate :class:`PipelineOutput` from a raw numeric prediction.

        Args:
            value: Predicted price.

        Returns:
            A :class:`PipelineOutput` instance wrapping *value*.
        """
        return cls(**{cls.get_target(): value})
