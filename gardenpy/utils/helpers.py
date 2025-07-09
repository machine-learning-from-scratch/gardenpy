r"""
**GardenPy helpers items.**

Minor helper items.

Contains:
    - :dict:`ansi`
    - :class:`Progress`
    - :func:`convert_time`
    - :func:`print_contributors`
"""

import sys

from .checkers import Params, ParamChecker

# common ansi formats
ansi: dict[str, str] = {
    'reset': '\033[0m',
    'black': '\033[30m',
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'white': '\033[37m',
    'bright_black': '\033[90m',
    'bright_red': '\033[91m',
    'bright_green': '\033[92m',
    'bright_yellow': '\033[93m',
    'bright_blue': '\033[94m',
    'bright_magenta': '\033[95m',
    'bright_cyan': '\033[96m',
    'bright_white': '\033[97m',
    'bold': '\033[1m',
    'dim': '\033[2m',
    'italic': '\033[3m',
    'underline': '\033[4m',
    'blinking': '\033[5m',
    'reverse': '\033[7m',
    'hidden': '\033[8m',
    'strikethrough': '\033[9m'
}


class Progress:
    r"""
    **Customizable progress bar.**

    Customizable progress bar, saving internals in state.
    """
    # default bar style
    _parameters: Params = Params(
        default={
            'length': 50,
            'left': f"{ansi['reset']}",
            'right': f"{ansi['reset']}",
            'completed': f"{ansi['green']}—",
            'uncompleted': f"{ansi['red']}—"
        },
        dtypes={'length': int, 'left': str, 'right': str, 'completed': str, 'uncompleted': str},
        vtypes={
            'length': lambda x: 0 < x,
            'left': lambda x: True,
            'right': lambda x: True,
            'completed': lambda x: True,
            'uncompleted': lambda x: True
        },
        ctypes={
            'length': lambda x: x,
            'left': lambda x: x,
            'right': lambda x: x,
            'completed': lambda x: x,
            'uncompleted': lambda x: x,
        }
    )

    def __init__(self, max_idx: int, *, bar_style: dict[str, any] | None = None, ikwiad: bool = False, **kwargs: any):
        r"""
        **Customizable progress bar.**

        Parameters:
            max_idx (int): Maximum bar index.
            bar_style (dict[str, any]): Bar visual style.
            ikwiad (bool), default = False: Turns off warning messages ("I know what I am doing" - ikwiad).
            **kwargs: Key-word bar visual style.

        Raises:
            TypeError: Invalid maximum index.
        """
        if not isinstance(max_idx, int) or max_idx <= 0:
            raise TypeError(
                "Invalid type: Maximum index must be a positive number. "
                f"Received type {type(max_idx)} of value {max_idx}."
            )
        self._max_idx: int = max_idx
        self._style: dict = self._get_style(style=bar_style, ikwiad=ikwiad, **kwargs)
        self._idx: int = 0

    @classmethod
    def _get_style(cls, style: dict[str, any], * , ikwiad: bool = False, **kwargs):
        # set checker
        checker = ParamChecker(
            prefix='bar style',
            parameters=cls._parameters,
            ikwiad=ikwiad
        )
        # return style
        return checker(params=style, **kwargs)

    def reset(self, show: bool = False, desc: str | None = None):
        r"""
        **Resets progress bar.**

        Resets the progress bar and internal index counter.
        Optionally shows the reset progress bar.

        Parameters:
            show (bool), default=False: Print reset progress bar.
            desc (str | None: Bar description. Alterable with each call.
        """
        self._idx = 0
        if show:
            sys.stdout.write(
                f"\r{ansi['reset']}{self._style['left']}"
                f"{ansi['reset']}{self._style['uncompleted'] * self._style['length']}"
                f"{ansi['reset']}{self._style['right']}{ansi['reset']}{desc or ''}{ansi['reset']}"
            )
            sys.stdout.flush()
        return None

    def __call__(self, desc: str | None = None) -> None:
        r"""
        **Displays progress bar.**

        Iterates the internal index counter with each call and displays the appropriate progress bar.
        Print a newline once the index counter reaches the maximum index.

        Parameters:
            desc (str | None): Bar description. Alterable with each call.
        """
        # completed progress
        comp = (self._idx + 1) / self._max_idx
        sys.stdout.write(
            f"\r{ansi['reset']}{self._style['left']}"
            f"{ansi['reset']}{self._style['completed'] * int(self._style['length'] * comp)}"
            f"{ansi['reset']}{self._style['uncompleted'] * (self._style['length'] - int(self._style['length'] * comp))}"
            f"{ansi['reset']}{self._style['right']}{ansi['reset']}{desc or ''}{ansi['reset']}"
        )
        sys.stdout.flush()
        if comp == 1:
            sys.stdout.write("\n")
        else:
            self._idx += 1
        return None


# todo: figure out import ordering and add _Tensor type
def visualize_cache(cache: list[dict[str, list | str | None] | None], itm_break: str = ' ') -> str:
    r"""
    **Turns raw cache into a color-visualized string.**

    Converted cache is not interactable; it is just a string.

    Parameters:
        cache (list[dict[str, _Tensor | list | str | None] | None]): Cache to be visualized.
        itm_break (str), default = '': Line break between cache items.

    Returns:
        str: Visualized cache
    """
    # initialize cache str
    cache_str = ""
    for itm in cache:
        if itm is None:
            # none item
            cache_str += f"{ansi['yellow']}None{ansi['reset']}{itm_break}"
            continue
        # normal item
        cache_str += "{"
        _, val = list(itm.items())[0]
        cache_str += f"{ansi['magenta']}{ansi['bold']}{val}{ansi['reset']} | ".replace("'", '')
        for key, val in list(itm.items())[1:-1]:
            cache_str += f"{ansi['cyan']}{key}{ansi['reset']}: {val} ".replace("'", '')
        key, val = list(itm.items())[-1]
        cache_str += f"{ansi['cyan']}{key}{ansi['reset']}: {val}".replace("'", '')
        cache_str += f"}}{itm_break}"
    return cache_str


def convert_time(seconds: float | int) -> str:
    r"""
    **Converts seconds to hours:minutes:seconds.**

    Parameters:
        seconds (float | int), 0 <= seconds: Number of seconds.

    Returns:
        str: Time in hours:minutes:seconds format.

    Raises:
        TypeError: Invalid second type.
    """
    # check seconds
    if not isinstance(seconds, (float, int)) or seconds < 0.0:
        raise TypeError(
            f"Invalid type: Seconds must be a positive number. "
            f"Received type {type(seconds)} of value {seconds}."
        )
    # calculate hours and minutes
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:01}:{minutes:02}:{seconds:02}"


def print_contributors(*, who: list[str] | None = None) -> None:
    r"""
    **Prints GardenPy contributors in alphabetical order.**

    The Machine Learning from Scratch team created GardenPy and our other resources with the help of many others.
    In this contributor printing function, we try to give thanks to the main resources and people that aided us in the
    creation of our project.
    At the same time, this function misses many vital contributors responsible for aiding us in the creation of our
    project, and we wish to thank anyone who helped us in any way.

    Parameters:
        who (list[str]): default = None: Contributor types.

    Raises:
        TypeError: Invalid contributor types datatype.
        ValueError: Invalid contributor type names.
    """
    # contributors
    contributors = {
        'programmers': [
            ["Christian SW Host-Madsen", "Punahou School CO '25", "<c.host.madsen25@gmail.com>"],
            ["Doyoung Kim", "Punahou School CO '25", "<dkim25@punahou.edu>"],
            ["Mason YY Morales", "Punahou School CO '25", "<mmorales25@punahou.edu>"],
            ["Isaac P Verbrugge", "Punahou School CO '25", "<isaacverbrugge@gmail.com>"],
            ["Derek S Yee", "Punahou School CO '25", "<dyee25@punahou.edu>"]
        ],
        'artists': [
            ["Kamalau Kimata", "Punahou School CO '25", "<kkimata25@punahou.edu>"]
        ],
        'thanks': [
            ['Justin Johnson', 'The University of Michigan'],
            ['The PyTorch Team', 'PyTorch'],
            ['Grant Sanderson', '3Blue1Brown'],
            ['Josh Starmer', 'StatQuest']
        ]
    }

    # clean who list
    contributor_types = ['programmers', 'artists', 'thanks']
    if isinstance(who, list):
        who = list(set(who))
    if not (isinstance(who, list) or who is None):
        raise TypeError(
            f"Invalid type: Contributor types must be a list. "
            f"Received type {type(who)}."
        )
    if who is not None and not all([(pers in contributor_types) for pers in who]):
        raise ValueError(
            f"Invalid contributor type: Invalid contributor detected in: {who}. "
            f"Choose from: {contributor_types}."
        )
    who = who or contributor_types

    # print contributors
    print(f"{ansi['bold']}{ansi['green']}GardenPy{ansi['reset']}")
    if 'programmers' in who:
        print(f"{ansi['bold']}Programmers{ansi['reset']}")
        for row in contributors['programmers']:
            print(
                "    {reset}{:<30} {white}{:<25}{reset} {bright_black}{:<20}{reset}"
                .format(row[0], row[1], row[2], **ansi)
            )
    if 'artists' in who:
        print(f"{ansi['bold']}Artists{ansi['reset']}")
        for row in contributors['artists']:
            print(
                "    {reset}{:<30} {white}{:<25}{reset} {bright_black}{:<20}{reset}"
                .format(row[0], row[1], row[2], **ansi)
            )
    if 'thanks' in who:
        print(f"{ansi['bold']}Special Thanks To{ansi['reset']}")
        for row in contributors['thanks']:
            print(f"    {ansi['reset']}{row[0]} {ansi['white']}from {row[1]}{ansi['reset']}")
    return None
