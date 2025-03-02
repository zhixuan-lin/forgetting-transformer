# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Various schedules. 

Most implementaion adapted from https://github.com/google-deepmind/optax/tree/main/optax/schedules
"""
from typing import Sequence, Callable, Optional
import numpy as np
import logging


Schedule = Callable[int, float]


def join_schedules(
    schedules: Sequence[Schedule], boundaries: Sequence[int]
) -> Schedule:
    """Sequentially apply multiple schedules.

    Args:
      schedules: A list of callables (expected to be optax schedules). Each
        schedule will receive a step count indicating the number of steps since
        the previous boundary transition.
      boundaries: A list of integers (of length one less than schedules) that
        indicate when to transition between schedules.
    Returns:
      schedule: A function that maps step counts to values.
    """

    def schedule(step: int) -> float:
        output = schedules[0](step)
        for boundary, schedule in zip(boundaries, schedules[1:]):
            if step >= boundary:
                output = schedule(step - boundary)
        return output

    return schedule


def constant_schedule(value: float):
    return lambda count: value

def one_minus_sqrt_schedule(
    init_value: int,
    end_value: int,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
    if transition_steps <= 0:
        logging.info(
            "A one minus sqrt schedule was set with a non-positive `transition_steps` "
            "value; this results in a constant schedule with value `init_value`."
        )
        return lambda count: init_value

    if transition_begin < 0:
        logging.info(
            "A one minus schedule was set with a negative `transition_begin` "
            "value; this will result in `transition_begin` falling back to `0`."
        )
        transition_begin = 0

    def schedule(count):
        count = int(np.clip(count - transition_begin, 0, transition_steps))
        frac =  count / transition_steps
        return (end_value - init_value) * (np.sqrt(frac)) + init_value

    return schedule

def polynomial_schedule(
    init_value: int,
    end_value: int,
    power: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
    """Constructs a schedule with polynomial transition from init to end value.

    Args:
      init_value: initial value for the scalar to be annealed.
      end_value: end value of the scalar to be annealed.
      power: the power of the polynomial used to transition from init to end.
      transition_steps: number of steps over which annealing takes place.
        The scalar starts changing at ``transition_begin`` steps and completes
        the transition by ``transition_begin + transition_steps`` steps.
        If ``transition_steps <= 0``, then the entire annealing process is
        disabled and the value is held fixed at ``init_value``.
      transition_begin: must be positive. After how many steps to start annealing
        (before this many steps the scalar value is held fixed at ``init_value``).

    Returns:
      schedule
        A function that maps step counts to values.
    """
    if transition_steps <= 0:
        logging.info(
            "A polynomial schedule was set with a non-positive `transition_steps` "
            "value; this results in a constant schedule with value `init_value`."
        )
        return lambda count: init_value

    if transition_begin < 0:
        logging.info(
            "A polynomial schedule was set with a negative `transition_begin` "
            "value; this will result in `transition_begin` falling back to `0`."
        )
        transition_begin = 0

    def schedule(count):
        count = int(np.clip(count - transition_begin, 0, transition_steps))
        frac = 1 - count / transition_steps
        return (init_value - end_value) * (frac**power) + end_value

    return schedule


def linear_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
    r"""Schedule with linear transition from ``init_value`` to ``end_value``.

    Args:
      init_value: initial value for the scalar to be annealed.
      end_value: end value of the scalar to be annealed.
      transition_steps: number of steps over which annealing takes place. The
        scalar starts changing at ``transition_begin`` steps and completes the
        transition by ``transition_begin + transition_steps`` steps. If
        ``transition_steps <= 0``, then the entire annealing process is disabled
        and the value is held fixed at ``init_value``.
      transition_begin: must be positive. After how many steps to start annealing
        (before this many steps the scalar value is held fixed at ``init_value``).

    Returns:
      schedule
        A function that maps step counts to values.
    """
    return polynomial_schedule(
        init_value=init_value,
        end_value=end_value,
        power=1,
        transition_steps=transition_steps,
        transition_begin=transition_begin,
    )


def cosine_decay_schedule(
    init_value: float,
    decay_steps: int,
    alpha: float = 0.0,
    exponent: float = 1.0,
) -> Schedule:
    r"""Returns a function which implements cosine learning rate decay.

  This schedule smoothly decreases the learning rate over a specified number of
  steps (``decay_steps``). The decay follows a cosine function, with an optional
  exponent to modify the decay curve. A minimum value (``alpha``) ensures the
  learning rate does not drop entirely to zero.

  More precisely, the learning rate at iteration :math:`t` is given by:

  .. math::
    \begin{cases}
      \frac{I (1 - \alpha)}{2}(1+\cos(\pi\,\frac{t}{T})^p) + \alpha\, 
      & \text{if } t \leq T \\
      I \alpha, & \text{if } t > T 
    \end{cases}

  where :math:`T` is the number of decay steps (``decay_steps``), :math:`p` is
  the ``exponent`` and :math:`I` is the initial value (``init_value``).

  References:
    Loshchilov et al., `SGDR: Stochastic Gradient Descent with Warm Restarts
    <https://arxiv.org/abs/1608.03983>`_, 2017

  Args:
    init_value: An initial value for the learning rate.
    decay_steps: Positive integer - the number of steps for which to apply
      the decay for.
    alpha: The minimum value of the multiplier used to adjust the
      learning rate. Defaults to 0.0.
    exponent:  The default decay is ``0.5 * (1 + cos(pi * t/T))``, where 
      ``t`` is the current timestep and ``T`` is the ``decay_steps``. The
      exponent modifies this to be ``(0.5 * (1 + cos(pi * t/T))) ** exponent``.
      Defaults to 1.0.

  Returns:
    schedule
      A function that maps step counts to values.
  """
    if not decay_steps > 0:
        raise ValueError(
            "The cosine_decay_schedule requires positive decay_steps, got"
            f" {decay_steps=}."
        )

    def schedule(count):
        count = np.minimum(count, decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * count / decay_steps))
        decayed = (1 - alpha) * cosine_decay**exponent + alpha
        return float(init_value * decayed)

    return schedule


def warmup_cosine_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0,
    exponent: float = 1.0,
) -> Schedule:
    r"""Linear warmup followed by cosine decay.

    Args:
      init_value: Initial value for the scalar to be annealed.
      peak_value: Peak value for scalar to be annealed at end of warmup.
      warmup_steps: Positive integer, the length of the linear warmup.
      decay_steps: Positive integer, the total length of the schedule. Note that
        this includes the warmup time, so the number of steps during which cosine
        annealing is applied is ``decay_steps - warmup_steps``.
      end_value: End value of the scalar to be annealed.
      exponent: Float. The default decay is ``0.5 * (1 + cos(pi t/T))``,
        where ``t`` is the current timestep and ``T`` is ``decay_steps``.
        The exponent modifies this to be ``(0.5 * (1 + cos(pi * t/T)))
        ** exponent``.
        Defaults to 1.0.

    Returns:
      schedule
        A function that maps step counts to values
    """
    alpha = 0.0 if peak_value == 0.0 else end_value / peak_value
    schedules = [
        linear_schedule(
            init_value=init_value,
            end_value=peak_value,
            transition_steps=warmup_steps,
        ),
        cosine_decay_schedule(
            init_value=peak_value,
            decay_steps=decay_steps - warmup_steps,
            alpha=alpha,
            exponent=exponent,
        ),
    ]
    return join_schedules(schedules, [warmup_steps])

def warmup_one_minus_sqrt_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    total_steps: int,
    anneal_steps: int,
    end_value: float = 0.0,
) -> Schedule:
    schedules = [
        linear_schedule(
            init_value=init_value,
            end_value=peak_value,
            transition_steps=warmup_steps,
        ),
        constant_schedule(
            value=peak_value
        ),
        one_minus_sqrt_schedule(
            init_value=peak_value,
            end_value=end_value,
            transition_steps=anneal_steps,
        ),
    ]
    return join_schedules(schedules, [warmup_steps, total_steps - anneal_steps])

def warmup_linear_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0,
) -> Schedule:
    r"""Linear warmup followed by cosine decay.

    Args:
      init_value: Initial value for the scalar to be annealed.
      peak_value: Peak value for scalar to be annealed at end of warmup.
      warmup_steps: Positive integer, the length of the linear warmup.
      decay_steps: Positive integer, the total length of the schedule. Note that
        this includes the warmup time, so the number of steps during which cosine
        annealing is applied is ``decay_steps - warmup_steps``.
      end_value: End value of the scalar to be annealed.
      exponent: Float. The default decay is ``0.5 * (1 + cos(pi t/T))``,
        where ``t`` is the current timestep and ``T`` is ``decay_steps``.
        The exponent modifies this to be ``(0.5 * (1 + cos(pi * t/T)))
        ** exponent``.
        Defaults to 1.0.

    Returns:
      schedule
        A function that maps step counts to values
    """
    alpha = 0.0 if peak_value == 0.0 else end_value / peak_value
    schedules = [
        linear_schedule(
            init_value=init_value,
            end_value=peak_value,
            transition_steps=warmup_steps,
        ),
        linear_schedule(
            init_value=peak_value,
            end_value=end_value,
            transition_steps=decay_steps - warmup_steps,
        ),
    ]
    return join_schedules(schedules, [warmup_steps])
