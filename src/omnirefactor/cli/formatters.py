from .imports import *


class CustomArgumentParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        formatter = self._get_formatter()
        for action_group in self._action_groups:
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)
            formatter.end_section()
        self._print_message(formatter.format_help(), file)


class CustomHelpFormatter(argparse.HelpFormatter):
    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = "usage: "
        prog = self._prog
        group_names = [f"[{group.title}]" for group in groups]
        usage_str = f"{prefix}{prog} {' '.join(group_names)}\n"
        return usage_str
