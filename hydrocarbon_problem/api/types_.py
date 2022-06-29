"""Unit definitions:
    Pressure: atm
    Temperature: degree C
    Molar flow: kmol/s
    Condenser/reboiler duty: Watt
    Distillate/reflux rate: kmol/s"""
from typing import NamedTuple, Tuple, Union

Array = Tuple[float, ...]

"""Types for flowsheet properties used to help specify the api between Aspen and Python"""
class PerCompoundProperty(NamedTuple):
    """Defines a type for each compound."""
    ethane: Union[float, str]
    propane: Union[float, str]
    isobutane: Union[float, str]
    n_butane: Union[float, str]
    isopentane: Union[float, str]
    n_pentane: Union[float, str]


class StreamSpecification(NamedTuple):
    """Complete specification of a stream"""
    temperature: float  # degrees C
    pressure: float  # atm
    molar_flows: PerCompoundProperty  # mol/s
    # T_condenser: float  # degrees C
    # T_reboiler: float  # degrees C


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
    condenser_temperature: float
    reboiler_temperature: float
    condenser_duty: float
    reboiler_duty: float
    diameter: Union[float, None]
    # molar_weight_per_stage: Array
    # vapor_flow_per_stage: Array
    # temperature_per_stage: Array


class ProductSpecification(NamedTuple):
    """Definition of a product stream"""
    # we could change this to a PerCompoundProperty if we want to have separate
    # purities for each compound.
    purity: float

# class RunSpecification(NamedTuple):
#     """Run status like duration and convergence"""
#     duration: float
#     converged: bool