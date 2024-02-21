from .wps_say_hello import SayHello
from .wps_sleep import Sleep
from .wps_drought import Drought

processes = [
    SayHello(),
    Drought(),
    Sleep(),
]
