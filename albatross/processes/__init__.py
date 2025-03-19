from .wps_drought import Drought
from .wps_say_hello import SayHello
from .wps_sleep import Sleep

processes = [
    SayHello(),
    Drought(),
    Sleep(),
]
