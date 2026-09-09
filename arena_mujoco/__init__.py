"""MuJoCo challenge arena with a deformable adhesive tape reference model."""

__version__ = "2.0.0"


def __getattr__(name):
    if name == "ArenaSimulation":
        from .runtime import ArenaSimulation
        return ArenaSimulation
    if name == "ArenaEnv":
        from .env import ArenaEnv
        return ArenaEnv
    raise AttributeError(name)
