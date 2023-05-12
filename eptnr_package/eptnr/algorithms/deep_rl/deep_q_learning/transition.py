from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'available_actions_next_state', 'reward'))
