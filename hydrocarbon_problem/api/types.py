from typing import NamedTuple


"""Types for flowsheet properties used to help specify the api between Aspen and Python"""
class PerCompoundProperty(NamedTuple):
    """Defines a type for each compound"""
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
    feed_stage_location: int
    n_stages: int
    reflux_ratio: float
    reboil_ratio: float
    pressure_drop_factor: float


class ColumnOutputSpecification(NamedTuple):
    condensor_duty: float
    reboiler_duty: float
