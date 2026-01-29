import io

from omnirefactor.cli.formatters import CustomArgumentParser, CustomHelpFormatter


def test_custom_help_formatter_usage_includes_groups(capsys):
    parser = CustomArgumentParser(
        prog="omni",
        formatter_class=CustomHelpFormatter,
    )
    parser.add_argument("--foo", help="foo option")
    train_group = parser.add_argument_group("train")
    train_group.add_argument("--train", action="store_true")
    eval_group = parser.add_argument_group("eval")
    eval_group.add_argument("--eval", action="store_true")

    out = parser.format_help()

    assert "usage: omni" in out
    assert "train:" in out
    assert "eval:" in out


def test_custom_argument_parser_print_help_uses_formatter():
    parser = CustomArgumentParser(
        prog="omni",
        formatter_class=CustomHelpFormatter,
    )
    parser.add_argument("--foo", help="foo option")
    train_group = parser.add_argument_group("train")
    train_group.add_argument("--train", action="store_true")
    eval_group = parser.add_argument_group("eval")
    eval_group.add_argument("--eval", action="store_true")

    buf = io.StringIO()
    parser.print_help(file=buf)
    out = buf.getvalue()
    assert "train:" in out
    assert "eval:" in out


def test_custom_help_formatter_format_usage():
    parser = CustomArgumentParser(
        prog="omni",
        formatter_class=CustomHelpFormatter,
    )
    parser.add_argument("--foo", help="foo option")
    train_group = parser.add_argument_group("train")
    train_group.add_argument("--train", action="store_true")
    eval_group = parser.add_argument_group("eval")
    eval_group.add_argument("--eval", action="store_true")

    usage = parser.format_usage()
    assert usage.startswith("usage: omni")
