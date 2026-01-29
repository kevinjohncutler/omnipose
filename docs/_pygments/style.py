from pygments.style import Style
from pygments.token import (
    Comment, Error, Keyword, Literal, Name, Operator, Punctuation,
    Generic, String, Text, Whitespace, 
)

class CustomStyle(Style):
    default_style = ""
    styles = {
        Comment: 'italic',
        # Comment: "#888888",
        # Comment.Multiline: "#888888",
        # Comment.Preproc: "#888888",
        # Comment.Single: "#888888",
        # Comment.Special: "#888888",

        Error: "#960050 bg:#1e0010",
        Keyword: "#04a3d8",
        Keyword.Constant: "#04a3d8",
        Keyword.Declaration: "#04a3d8",
        Keyword.Namespace: "#f0147a",
        Keyword.Pseudo: "#01aecf",
        Keyword.Reserved: "#6322f9",
        Keyword.Type: "#2263f9",
        Literal: "#06dd9c",
        Literal.Date: "#c8b600",
        Literal.Number: "#06dd9c",
        Literal.Number.Float: "#06dd9c",
        Literal.Number.Hex: "#06dd9c",
        Literal.Number.Integer: "#06dd9c",
        Literal.Number.Oct: "#06dd9c",
        Literal.String: "#c8b600",
        Literal.String.Backtick: "#c8b600",
        Literal.String.Char: "#c8b600",
        Literal.String.Doc: "#c8b600",
        Literal.String.Double: "#c8b600",
        Literal.String.Escape: "#06dd9c",
        Literal.String.Heredoc: "#c8b600",
        Literal.String.Interpol: "#c8b600",
        Literal.String.Other: "#c8b600",
        Literal.String.Regex: "#c8b600",
        Literal.String.Single: "#c8b600",
        Literal.String.Symbol: "#c8b600", 
        Name.Variable.Instance: "#0de989",
        Name.Variable.Global: "#ff4040",
        Generic.Deleted: "#f0147a",
        Operator.Word: "#6322f9",
        Operator: "#888888",
        # Name.Tag: "#ff0000",
        Punctuation: "#888888",
        Name.Function: "#ff0000",
        Name: 'bold',

    }

# .highlight .c { color: var(--color-foreground-secondary)} /* Comment */
# .highlight .err { color: #960050; background-color: #1e0010 } /* Error */
# .highlight .k { color: #04a3d8 } /* Keyword */
# .highlight .l { color: #f27616 } /* Literal */
# .highlight .n { color: var(--color-foreground-primary) } /* Name */
# .highlight .o { color: #f0147a } /* Operator */
# .highlight .p { color: var(--color-foreground-primary) } /* Punctuation */
# .highlight .cm { color: var(--color-foreground-secondary) } /* Comment.Multiline */
# .highlight .cp { color: var(--color-foreground-secondary) } /* Comment.Preproc */
# .highlight .c1 { color: var(--color-foreground-secondary) } /* Comment.Single */
# .highlight .cs { color: var(--color-foreground-secondary) } /* Comment.Special */
# .highlight .ge { font-style: italic } /* Generic.Emph */
# .highlight .gs { font-weight: bold } /* Generic.Strong */
# .highlight .kc { color: #04a3d8 } /* Keyword.Constant */
# .highlight .kd { color: #04a3d8 } /* Keyword.Declaration */
# .highlight .kn { color: #f0147a } /* Keyword.Namespace */
# .highlight .kp { color: #04a3d8 } /* Keyword.Pseudo */
# .highlight .kr { color: #04a3d8 } /* Keyword.Reserved */
# .highlight .kt { color: #04a3d8 } /* Keyword.Type */
# .highlight .ld { color: #c8b600 } /* Literal.Date */
# .highlight .m { color: #f27616 } /* Literal.Number */
# .highlight .s { color: #c8b600 } /* Literal.String */
# .highlight .na { color: #aecf01 } /* Name.Attribute */
# .highlight .nb { color: var(--color-foreground-primary) } /* Name.Builtin */
# .highlight .nc { color: #aecf01 } /* Name.Class */
# .highlight .no { color: #04a3d8 } /* Name.Constant */
# .highlight .nd { color: #aecf01 } /* Name.Decorator */
# .highlight .ni { color: var(--color-foreground-primary) } /* Name.Entity */
# .highlight .ne { color: #aecf01 } /* Name.Exception */
# .highlight .nf { color: #aecf01 } /* Name.Function */
# .highlight .nl { color: var(--color-foreground-primary) } /* Name.Label */
# .highlight .nn { color: var(--color-foreground-primary) } /* Name.Namespace */
# .highlight .nx { color: #aecf01 } /* Name.Other */
# .highlight .py { color: var(--color-foreground-primary) } /* Name.Property */
# .highlight .nt { color: #f0147a } /* Name.Tag */
# .highlight .nv { color: var(--color-foreground-primary) } /* Name.Variable */
# .highlight .ow { color: #f0147a } /* Operator.Word */
# .highlight .w { color: var(--color-foreground-primary) } /* Text.Whitespace */
# .highlight .mf { color: #f27616 } /* Literal.Number.Float */
# .highlight .mh { color: #f27616 } /* Literal.Number.Hex */
# .highlight .mi { color: #f27616 } /* Literal.Number.Integer */
# .highlight .mo { color: #f27616 } /* Literal.Number.Oct */
# .highlight .sb { color: #c8b600 } /* Literal.String.Backtick */
# .highlight .sc { color: #c8b600 } /* Literal.String.Char */
# .highlight .sd { color: #c8b600 } /* Literal.String.Doc */
# .highlight .s2 { color: #c8b600 } /* Literal.String.Double */
# .highlight .se { color: #f27616 } /* Literal.String.Escape */
# .highlight .sh { color: #c8b600 } /* Literal.String.Heredoc */
# .highlight .si { color: #c8b600 } /* Literal.String.Interpol */
# .highlight .sx { color: #c8b600 } /* Literal.String.Other */
# .highlight .sr { color: #c8b600 } /* Literal.String.Regex */
# .highlight .s1 { color: #c8b600 } /* Literal.String.Single */
# .highlight .ss { color: #c8b600 } /* Literal.String.Symbol */
# .highlight .bp { color: var(--color-foreground-primary) } /* Name.Builtin.Pseudo */
# .highlight .vc { color: var(--color-foreground-primary) } /* Name.Variable.Class */
# .highlight .vg { color: var(--color-foreground-primary) } /* Name.Variable.Global */
# .highlight .vi { color: var(--color-foreground-primary) } /* Name.Variable.Instance */
# .highlight .il { color: #f27616 } /* Literal.Number.Integer.Long */

# /* .highlight .gh { } Generic Heading & Diff Header */
# .highlight .gu { color: var(--color-foreground-secondary); } /* Generic.Subheading & Diff Unified/Comment? */
# .highlight .gd { color: #f0147a; } /* Generic.Deleted & Diff Deleted */
# .highlight .gi { color: #aecf01; } Generic.Inserted & Diff Inserted