from typing import NamedTuple


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
    feed_stage_location: int
    n_stages: int
    reflux_ratio: float  # between 0 and 1
    reboil_ratio: float  # between 0 and 1
    pressure_drop: float  # atm


class ColumnOutputSpecification(NamedTuple):
    """All relevant output information from the simulated column (besides output stream info)."""
    condensor_duty: float
    reboiler_duty: float
