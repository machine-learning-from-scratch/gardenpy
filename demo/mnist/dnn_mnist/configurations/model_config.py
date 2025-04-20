from easydict import EasyDict as Edict

########################################################################################################################

# model architecture
architecture: Edict = Edict()

# sizes
architecture.layer_sizes = [784, 256, 128, 64, 10]

# thetas
architecture.weights = Edict()
architecture.weights.methods = ['kaiming', 'kaiming', 'kaiming', 'kaiming']
architecture.weights.hyperparams = [
    {'beta': 1e-02, 'mu': 0.0, 'sigma': 1.0, 'kappa': 1.0},
    {'beta': 1e-02, 'mu': 0.0, 'sigma': 1.0, 'kappa': 1.0},
    {'beta': 1e-02, 'mu': 0.0, 'sigma': 1.0, 'kappa': 1.0},
    {'beta': 1e-02, 'mu': 0.0, 'sigma': 1.0, 'kappa': 1.0}
]
architecture.biases = Edict()
architecture.biases.methods = ['uniform', 'uniform', 'uniform', 'uniform']
architecture.biases.hyperparams = [{'kappa': 0.0}, {'kappa': 0.0}, {'kappa': 0.0}, {'kappa': 0.0}]

# activators
architecture.activators = Edict()
architecture.activators.methods = ['lrelu', 'lrelu', 'lrelu', 'softmax']
architecture.activators.hyperparams = [{'beta': 1e-02}, {'beta': 1e-02}, {'beta': 1e-02}, {'beta': 1e-02}, None]

########################################################################################################################

# model peripherals
peripherals: Edict = Edict()

# criterion
peripherals.criterion = Edict()
peripherals.criterion.method = 'centropy'
peripherals.criterion.hyperparams = None

# optimizer
peripherals.optimizer = Edict()
peripherals.optimizer.method = 'adam'
peripherals.optimizer.hyperparams = {
    'alpha': 1e-3,
    'lambda_d': 0.0,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-10,
    'ams': False
}

########################################################################################################################

__all__ = ['architecture', 'peripherals']
