"""Render the installed PerTurbo command's help on the CLI reference page."""

import subprocess
import sysconfig
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive


class PerTurboHelp(Directive):
    def run(self):
        command = Path(sysconfig.get_path("scripts")) / "perturbo"
        help_text = subprocess.check_output([str(command), "--help"], text=True)
        return [nodes.literal_block(text=help_text.rstrip())]


def setup(app):
    app.add_directive("perturbo-help", PerTurboHelp)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
