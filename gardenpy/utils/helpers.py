r"""
**Helpers functions.**

Contains:
    - :dict:`ansi`
    - :func:`progress`
    - :func:`convert_time`
    - :func:`slow_print`
    - :func:`print_contributors`
"""

import sys
import time

# common ansi formats
ansi = {
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


def progress(idx: int, max_idx: int, *, desc: str | None = None, b_len: int = 50, b_type: int = 0) -> None:
    r"""
    **Customizable progress bar.**

    Parameters:
        idx (int), 0 < idx: Current loop index.
        max_idx (int), 0 < max_idx: Maximum loop index.
        desc (str | None), default = None: Progress bar description.
        b_len (int), default = 50, 0 < b_len: Bar length.
        b_type (int), default = 0: Visual bar type.

    Raises:
        TypeError: Incorrect parameter types.
    """
    # todo: errors
    # check for errors
    if not (isinstance(idx, int) and -1 <= idx):
        raise TypeError("Referenced loop index must be a positive integer.")
    if not (isinstance(max_idx, int) and 0 < max_idx):
        raise TypeError("Maximum loop index must be a positive integer.")
    if not (isinstance(b_len, int) and 0 < b_len):
        raise TypeError("Bar length must be a positive integer.")
    # completed progress
    completed = (idx + 1) / max_idx
    # make progress bar
    if b_type == 0:
        sys.stdout.write(
            f"\r{ansi['reset']}"
            f"[{ansi['green']}{'—' * int(b_len * completed)}"
            f"{ansi['red']}{'—' * (b_len - int(b_len * completed))}"
            f"{ansi['reset']}]  {desc or ''}"
        )
    elif b_type == 1:
        sys.stdout.write(
            f"\r{ansi['bright_black']}"
            f"[{ansi['reset']}{'—' * int(b_len * completed)}"
            f"{ansi['reset']}{' ' * (b_len - int(b_len * completed))}"
            f"{ansi['bright_black']}]{ansi['reset']}  {desc or ''}"
        )
    elif b_type == 2:
        sys.stdout.write(
            f"\r{ansi['reset']}"
            f"|{'█' * int(b_len * completed)}"
            f"{' ' * (b_len - int(b_len * completed))}"
            f"{ansi['reset']}|  {desc or ''}"
        )
    else:
        sys.stdout.write(
            f"\r{ansi['reset']}"
            f"[{'—' * int(b_len * completed)}"
            f"{' ' * (b_len - int(b_len * completed))}"
            f"{ansi['reset']}]  {desc or ''}"
        )
    sys.stdout.flush()
    if completed == 1:
        sys.stdout.write("\n")
    return None


def convert_time(seconds: float | int) -> str:
    r"""
    **Converts seconds to hours:minutes:seconds.**

    Args:
        seconds (float | int), 0 < seconds: Number of seconds.

    Returns:
        str: Time in hours:minutes:seconds format.

    Raises:
        TypeError: Incorrect parameter types.
    """
    # check for errors
    if not (isinstance(seconds, (float, int)) and 0 <= seconds):
        raise TypeError("Converted seconds must be a positive real number.")
    # calculate hours and minutes
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    # return time
    return f"{hours:01}:{minutes:02}:{seconds:02}"


def slow_print(text: str, *, delay: float | int = 0.05) -> None:
    r"""
    **Prints text with delay.**

    Args:
        text (str): Text to print.
        delay (float | int), default = 0.05, 0 < delay: Delay between characters in seconds.

    Raises:
        TypeError: If parameters are of the wrong type.
    """
    # check for errors
    if not isinstance(text, str):
        raise TypeError("'text' must be a str")
    if not (isinstance(delay, (float, int)) and 0 < delay):
        raise TypeError("'delay' must be a positive float or integer")
    # print text
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    return None


def print_contributors(*, who: list[str] | None = None) -> None:
    r"""
    **Prints GardenPy contributors in alphabetical order.**

    The Machine Learning from Scratch team created GardenPy and our _other resources with the help of many others.
    In this contributor printing function, we try to give thanks to the main resources and people that aided us in the
    creation of our project.
    At the same time, this function misses many vital contributors responsible for aiding us in the creation of our
    project, and we wish to thank anyone who helped us in any way.

    Parameters:
        who: default = all: Type of contributors to print.

    Raises:
        TypeError: Incorrect parameter types.
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
        raise TypeError("'who' must be a list")
    if who is not None and not all([(pers in contributor_types) for pers in who]):
        raise ValueError(
            f"Invalid contributor type detected in: {who}\n"
            f"Choose from: {contributor_types}"
        )
    who = who or contributor_types

    # print contributors
    print(f"{ansi['bold']}{ansi['green']}GardenPy{ansi['reset']}")
    if 'programmers' in who:
        print(f"{ansi['bold']}Programmers{ansi['reset']}", end='\n')
        for row in contributors['programmers']:
            print(
                "    {reset}{:<30} {white}{:<25}{reset} {bright_black}{:<20}{reset}"
                .format(row[0], row[1], row[2], **ansi)
            )
    if 'artists' in who:
        print(f"{ansi['bold']}Artists{ansi['reset']}", end='\n')
        for row in contributors['artists']:
            print(
                "    {reset}{:<30} {white}{:<25}{reset} {bright_black}{:<20}{reset}"
                .format(row[0], row[1], row[2], **ansi)
            )
    if 'thanks' in who:
        print(f"{ansi['bold']}Special Thanks To{ansi['reset']}", end='\n')
        for row in contributors['thanks']:
            print(f"    {ansi['reset']}{row[0]} {ansi['white']}from {row[1]}{ansi['reset']}")
    return None
