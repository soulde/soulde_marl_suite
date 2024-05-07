from dataclasses import dataclass
import re


@dataclass
class AgentParams:
    """
    parameters for agent training
    """
    '''
    General parameters
    '''
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    gamma: float = 0.9
    tau: float = 0.001
    policy_update_frequency: int = 2
    target_network_frequency: int = 2
    max_grad_norm: float = 0.01

    """
    PPO parameters
    """
    ent_param: float = 0.5
    vf_param: float = 0.5
    clip_param: float = 1

    '''
    TD3 parameters
    '''
    noise_param: float = 0.1

    def from_config(self, config):
        for i in self.__dir__():
            if not re.match('__.+__', i) and not i == 'from_config':
                try:
                    self.__dict__[i] = config[str(i)]
                except KeyError:
                    pass


if __name__ == '__main__':
    params = AgentParams().from_config(None)
