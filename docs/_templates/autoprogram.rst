{% for group in parser._action_groups %}
{{ group.title }}
{{ "=" * group.title|length }}

{{ group.description }}

   {% for action in group._group_actions %}
   - {{ action.option_strings|join(', ') }}: {{ action.help|e }}
   {% endfor %}
{% endfor %}
