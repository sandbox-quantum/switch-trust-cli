from dataclasses import dataclass

from dataclasses_json import dataclass_json

from switch_trust.eval.core.eval.evaluation import Evaluation
from switch_trust.eval.core.eval.evaluation_garak_probe import GarakProbeEvaluation
from switch_trust.eval.core.eval.evaluation_multi import MultiEvaluation
from switch_trust.eval.core.optional_deps import require_garak


@dataclass_json
@dataclass
class GarakModuleEvaluation(MultiEvaluation):
    """Evaluates all probes in a garak module (e.g. ``"apikey"``)."""

    module_name: str | None = None
    probe_names: list[str] | None = None

    def __init__(
        self,
        module_name: str,
        probe_names: list[str] | None = None,
    ):
        super().__init__()
        self.module_name = module_name
        self.probe_names = probe_names

    async def get_children(self) -> list[Evaluation]:
        if not self.module_name:
            raise ValueError("module_name is required")

        if self.probe_names:
            names = self.probe_names
        else:
            prefix = f"probes.{self.module_name}."
            names = [
                name
                for name, active in require_garak()._plugins.enumerate_plugins("probes")
                if name.startswith(prefix) and active
            ]

        if not names:
            raise ValueError(f"no active probes found for module '{self.module_name}'")

        return [GarakProbeEvaluation(probe_name=name) for name in names]
