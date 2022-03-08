from typing import NamedTuple, Tuple

Array = Tuple[float, ...]

"""Types for flowsheet properties used to help specify the api between Aspen and Python"""
class PerCompoundProperty(NamedTuple):
    """Defines a type for each compound."""
    ethane: float
    propane: float
    isobutane: float
    n_butane: float
    isopentane: float
    n_pentane: float


class StreamSpecification(NamedTuple):
    """Complete specification of a stream"""
    temperature: float  # degrees C
    pressure: float  # atm
    molar_flows: PerCompoundProperty  # mol/s


class ColumnInputSpecification(NamedTuple):
    """Specification of the column (which along with the input stream, fully specifies the
    flowsheet)."""
    n_stages: int
    feed_stage_location: int
    reflux_ratio: float  # between 0 and infinity
    reboil_ratio: float  # between 0 and infinity
    condensor_pressure: float  # atm


class ColumnOutputSpecification(NamedTuple):
    """All relevant output information from the simulated column (besides output stream info)."""
    condensor_duty: float
    reboiler_duty: float
    molar_weight_per_stage: Array
    vapor_flow_per_stage: Array
    temperature_per_stage: Array


class ProductSpecification(NamedTuple):
    """Definition of a product stream"""
    # we could change this to a PerCompoundProperty if we want to have separate
    # purities for each compound.
    purity: float
