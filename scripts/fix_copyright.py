#!/usr/bin/env python3
# coding: utf-8

###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import datetime
import io
import os
import subprocess as sp
import sys

import click

current_year = datetime.date.today().year

formats = {"cpp": ("*", " *", "/*", " */"), "script": ("#", "#", "#", "")}


def prepare_copyright(created, modified, formatting):
    linefill, prefix, comment, close = formatting
    copyright = [
        " Copyright (C) {dates} Habana Labs, Ltd. an Intel Company",
        " All Rights Reserved.",
        "",
        " Unauthorized copying of this file or any element(s) within it, via any medium",
        " is strictly prohibited.",
        " This file contains Habana Labs, Ltd. proprietary and confidential information",
        " and is subject to the confidentiality and license agreements under which it",
        " was provided.",
        "",
    ]
    dates = str(modified) if created == modified else f"{created}-{modified}"
    cpr = [(prefix + copyright[0].format(dates=dates))]
    cpr += [f"{prefix}{c}" for c in copyright[1:]]
    maxl = max(len(c) for c in cpr)
    bar = linefill * (maxl - len(prefix))
    cpr = [comment + bar] + cpr + [prefix + bar]
    if close:
        cpr.append(close)
    cpr.append("")
    return "\n".join(cpr)


def propose_formatting(f):
    _, ext = os.path.splitext(f)
    formatting = formats["cpp"] if ext in (".c", ".cpp", ".h", ".hpp") else formats["script"]
    created = sp.check_output(
        f"git log --follow --format=%cs --date default {f} | tail -1", shell=True, encoding="ascii"
    )
    created = int(created[:4])
    modified = current_year
    cpr = prepare_copyright(created, modified, formatting)
    return cpr


def prefix_copyright(f):
    contents = list(open(f).readlines())
    cpr = propose_formatting(f)
    with open(f, "w") as out:
        out.write(cpr)
        out.write("".join(contents))


class PatchError(Exception):
    def __init__(self, file, what):
        super().__init__(f"{file} ", what)
        self.file = file


class NoCopyrightError(PatchError):
    pass


def _patch_file(f):
    try:
        lines = list(open(f).readlines())
    except FileNotFoundError:
        click.echo(f"NOT FOUND {f}", err=True)
        raise PatchError(f, "File not found")
    contents = lines[:]
    result = 0
    while contents[0] == "\n":  # remove blank lines from the top
        contents = contents[1:]
        result = 1
    script_header = list()  # shebang or other stuff that goes above the copyright definition.
    for l in contents[:3]:
        if l[:2] == "#!":
            script_header.append(l)
        elif l == "# coding: utf-8\n":
            script_header.append(l)
        elif l == "\n":
            script_header.append(l)
        else:
            break
    contents = contents[len(script_header) :]

    if "#" in contents[0]:
        formatting = formats["script"]
    elif "/*" in contents[0] or "//" in contents[0]:
        formatting = formats["cpp"]
    else:
        raise PatchError(f, f"unknown header in file")
    i = contents[1].split(" ")

    if not "Habana" in contents[1]:
        raise NoCopyrightError(f, f"Unexpected start of file {f}, not a Habana header")
    try:
        p = i.index("(C)")
    except ValueError:
        raise PatchError(f, f"Unexpected start of file {f}, expected copyright sign '(C)'")
    years = i[p + 1]
    if "-" in years:
        created, modified = years.split("-")
    elif "," in years:
        created, modified = years.split(",")
    else:
        created, modified = years, years

    try:
        created, modified = int(created), int(modified)
    except ValueError:
        raise PatchError(f, "Failed to parse either '{created}' or '{modified}' as a year number")
    created, modified = created if created > 2000 else created + 2000, modified if modified > 2000 else modified + 2000
    modified = current_year
    end_of_cpr = None
    for i, l in enumerate(contents[5:20]):
        if l == "\n":
            end_of_cpr = i + 5
            break
        if "*/" in l or "####################" in l:
            end_of_cpr = i + 6
            break
    if end_of_cpr is None:
        raise PatchError(f, f"Failed to identify end of copyright header in first 20 lines of {f}")
    cpr = prepare_copyright(created, modified, formatting)
    prev = "".join(contents[:end_of_cpr])
    if cpr == prev:
        return result
    with open(f, "w") as out:
        out.write("".join(script_header))
        out.write(cpr)
        out.write("".join(contents[end_of_cpr:]))
    return 1


def patch_file(f, prefix, verbose):
    try:
        return _patch_file(f)
    except FileNotFoundError:
        click.echo(f"NOT FOUND {f}", err=True)
        return
    except NoCopyrightError as e:
        if prefix:
            prefix_copyright(e.file)
            return 1
        else:
            fname = e.file, e if verbose else ""
            click.echo(f"FAILED {fname}", err=True)
        return 2
    except PatchError as e:
        click.echo(f"FAILED {e}", err=True)
        return 2


def sp_output_lines(cmd):
    proc = sp.Popen(cmd, stdout=sp.PIPE)
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
        yield line.strip()


@click.command()
@click.argument("file_names", nargs=-1)
@click.option(
    "--prefix",
    is_flag=True,
    default=False,
    help="Prefix guessed copyright header to files that appear to have no header at all.",
)
@click.option("--verbose", is_flag=True, default=False, help="Print verbose error.")
@click.option(
    "--git",
    type=click.Choice(["staged", "commited"]),
    default=None,
    help="Process git-versioned files: either staged or commited (cwd must be repo). If set FILE_NAMES is ignored.",
)
def patch_files(file_names, prefix, verbose, git):
    """
    Simple, stupid and effective tool to help with copyright header update.
    \b
    (1) It will open file(s) and assume that Habana copyright is in at the top.
    (2) It will capture the creation year.
    (3) It will then update the file with the new Habana copyright header with year range starting with the original creation year and current year.

    Don't fully trust this tool. Make sure to always review that the updates were correct.

    Examples:

    \b
    Process files staged from commit, write missing copyright headers.
        fix_copyright.py --prefix --git=staged

    \b
    Same as above but don't try fixing the missing headers, only complain.
        git diff-tree --no-commit-id --name-only HEAD -r | fix_copyright.py --verbose -



    FILE_NAMES is the list of files to process, or - to read file names line-by-line from stdin.
    """
    file_names = set(file_names)
    error_code = 0
    if git == "staged":
        file_names = sp_output_lines(["git", "diff", "--name-only", "--cached"])
    elif git == "commited":
        file_names = sp_output_lines(["git", "diff-tree", "--no-commit-id", "--name-only", "HEAD", "-r"])
    for name in file_names:
        if name == "-":
            for line in sys.stdin:
                error_code = max(error_code, patch_file(line.strip(), prefix, verbose))
        else:
            error_code = max(error_code, patch_file(name, prefix, verbose))
    sys.exit(error_code)


if __name__ == "__main__":
    patch_files()
