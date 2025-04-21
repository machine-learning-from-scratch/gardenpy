r"""
**MNIST DNN model config.**

Contains:
    - :class:`architecture`
    - :class:`peripherals`
"""

from dataclasses import dataclass

########################################################################################################################


# thetas config
@dataclass
class WeightsConfig:
    methods: list[str]
    hyperparams: list[dict[str, float | int | bool] | None]


@dataclass
class BiasesConfig:
    methods: list[str]
    hyperparams: list[dict[str, float | int | bool] | None]


# activators config
@dataclass
class ActivatorsConfig:
    methods: list[str]
    hyperparams: list[dict[str, float | int | bool] | None]


# criterion config
@dataclass
class CriterionConfig:
    method: str
    hyperparams: dict[str, float | int | bool] | None


# optimizer config
@dataclass
class OptimizerConfig:
    method: str
    hyperparams: dict[str, float | int | bool] | None


# general configs
@dataclass
class ArchitectureConfig:
    layer_sizes: list[int]
    weights: WeightsConfig
    biases: BiasesConfig
    activators: ActivatorsConfig


@dataclass
class PeripheralsConfig:
    criterion: CriterionConfig
    optimizer: OptimizerConfig


########################################################################################################################

architecture: ArchitectureConfig = ArchitectureConfig(
    layer_sizes=[784, 256, 128, 64, 10],
    weights=WeightsConfig(
        methods=['kaiming' for _ in range(4)],
        hyperparams=[{'beta': 1e-02, 'mu': 0.0, 'sigma': 1.0, 'kappa': 1.0} for _ in range(4)]
    ),
    biases=BiasesConfig(
        methods=['uniform' for _ in range(4)],
        hyperparams=[{'kappa': 0.0} for _ in range(4)]
    ),
    activators=ActivatorsConfig(
        methods=['lrelu', 'lrelu', 'lrelu', 'softmax'],
        hyperparams=[{'beta': 1e-02}, {'beta': 1e-02}, {'beta': 1e-02}, None]
    )
)

peripherals: PeripheralsConfig = PeripheralsConfig(
    criterion=CriterionConfig(
        method='centropy',
        hyperparams=None
    ),
    optimizer=OptimizerConfig(
        method='adam',
        hyperparams={
            'alpha': 1e-3,
            'lambda_d': 1e-4,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-10,
            'ams': False
        }
    )
)

########################################################################################################################

__all__: list[str] = ['architecture', 'peripherals']
