"""Energy and power prediction package for the E1-1plus AC unit.

The interface mirrors :mod:`simu.energy` so callers (in particular
``simu.air_conditioner.AirConditionerSimulator``) can swap in the E1-1plus
model with no code changes other than passing the alternate ``EnergyModel``.

Typical use::

    from simu.energy_e11plus.model import EnergyModel
    model = EnergyModel(mode="制冷")
    power = model.predict_power(controls)   # controls = [freq, eev, fan_out]
"""
