# Omnirefactor GUI port notes

This folder is the standalone PyWebView GUI ported from the main omnipose repo.
When re-syncing from the main repo, these are the key steps we followed:

- Copy `gui/web/` assets from the main repo into `omnirefactor/src/omnirefactor/gui/web/`.
- Move `gui/examples/pywebview_image_viewer.py` into `omnirefactor/src/omnirefactor/gui/app.py`.
- Move `gui/examples/sample_image.py` into `omnirefactor/src/omnirefactor/gui/sample_image.py`.
- Update module references in `app.py` (entrypoints, imports) to use `omnirefactor.gui`.
- Ensure `WEB_DIR` points at `omnirefactor/src/omnirefactor/gui/web`.
- Update any model/utility imports in `app.py` to use omnirefactor APIs.
- Package the web assets in `omnirefactor/setup.py` via `package_data` so wheels include them.
- Ensure CLI entrypoints call `omnirefactor.gui.app:main` (via `omnirefactor.__main__`).

Recommended re-sync workflow:

1. Pull new changes in the main repo `gui/` folder.
2. Copy over `gui/web/` and the two example files again.
3. Re-apply the import/path updates above.
4. Build a wheel and confirm `gui/web` assets are present.
