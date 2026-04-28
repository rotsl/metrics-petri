from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import ipywidgets as widgets
from IPython.display import Markdown, display


@dataclass
class ImageSelectionUI:
    state: dict[str, list[str]] = field(default_factory=lambda: {"selected_names": [], "uploaded_names": []})
    engine_widget: widgets.Dropdown | None = None
    generic_model_widget: widgets.Text | None = None
    generic_key_widget: widgets.Password | None = None
    upload_widget: widgets.FileUpload | None = None
    folder_accordion: widgets.Accordion | None = None
    selection_status: widgets.HTML | None = None
    selection_panel: widgets.VBox | None = None
    refresh_selector: Callable[[], None] | None = None
    apply_selection: Callable[[object | None], None] | None = None


def build_image_selection_ui(
    input_dir: Path,
    default_generic_model: str,
    initial_generic_key: str,
    visible_input_images: Callable[[Path], list[Path]],
    display_image_name: Callable[[Path], str],
) -> ImageSelectionUI:
    state: dict[str, list[str]] = {"selected_names": [], "uploaded_names": []}
    ui = ImageSelectionUI(state=state)

    ui.engine_widget = widgets.Dropdown(
        options=[("Local", "local"), ("Gemini API + Local", "gemini")],
        value="local",
        description="Engine",
        layout=widgets.Layout(width="340px"),
    )

    ui.generic_model_widget = widgets.Text(
        value=default_generic_model,
        description="Generic model",
        layout=widgets.Layout(width="420px"),
    )

    ui.generic_key_widget = widgets.Password(
        value=initial_generic_key,
        description="Generic key",
        layout=widgets.Layout(width="420px"),
    )

    ui.folder_accordion = widgets.Accordion(layout=widgets.Layout(width="100%"))
    folder_selectors: dict[str, widgets.SelectMultiple] = {}
    ui.selection_status = widgets.HTML("<b>Selected 0 image(s)</b>")

    def refresh_selector() -> None:
        available = visible_input_images(input_dir)
        grouped: dict[str, list[Path]] = {}
        for path in available:
            try:
                folder = path.relative_to(input_dir).parent.as_posix()
            except ValueError:
                folder = "."
            grouped.setdefault(folder, []).append(path)

        children = []
        titles = []
        previous_selection = set(ui.state.get("selected_names", []))
        folder_selectors.clear()

        for folder, paths in sorted(grouped.items(), key=lambda item: item[0]):
            options = [(display_image_name(path), display_image_name(path)) for path in paths]
            selector = widgets.SelectMultiple(
                options=options,
                value=tuple(name for _, name in options if name in previous_selection),
                layout=widgets.Layout(width="100%", height="150px"),
            )
            folder_selectors[folder] = selector
            children.append(selector)
            titles.append("input_images" if folder == "." else folder)

        ui.folder_accordion.children = tuple(children)
        for index, title in enumerate(titles):
            folder_key = "." if title == "input_images" else title
            ui.folder_accordion.set_title(index, f"{title} ({len(grouped[folder_key])} image(s))")

        selected = sorted({name for selector in folder_selectors.values() for name in selector.value})
        ui.state["selected_names"] = selected
        ui.selection_status.value = f"<b>Selected {len(selected)} image(s)</b>"

    def apply_selection(_=None) -> None:
        selected = sorted({name for selector in folder_selectors.values() for name in selector.value})
        ui.state["selected_names"] = selected
        ui.selection_status.value = f"<b>Selected {len(selected)} image(s)</b>"

    def on_image_upload(change) -> None:
        uploaded = change["new"]
        if not uploaded:
            return

        existing = {path.name for path in input_dir.glob("*") if path.is_file()}
        saved: list[str] = []
        for filename, payload in uploaded.items():
            target = input_dir / filename
            stem = target.stem
            suffix = target.suffix
            counter = 1
            while target.name in existing or target.exists():
                target = input_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            target.write_bytes(payload["content"])
            existing.add(target.name)
            saved.append(target.name)

        ui.state["uploaded_names"] = sorted(set(ui.state["uploaded_names"]).union(saved))
        display(Markdown(f"**Uploaded {len(saved)} image(s) into `{input_dir}`.**"))
        refresh_selector()

    ui.upload_widget = widgets.FileUpload(
        accept="image/*",
        multiple=True,
        description="Upload images",
    )
    ui.upload_widget.observe(on_image_upload, names="value")

    apply_button = widgets.Button(description="Apply selection", button_style="primary")
    apply_button.on_click(apply_selection)

    controls = widgets.VBox(
        [
            ui.engine_widget,
            ui.gemini_model_widget,
            ui.gemini_key_widget,
            ui.upload_widget,
            ui.selection_status,
            apply_button,
        ],
        layout=widgets.Layout(gap="0.5rem"),
    )

    ui.selection_panel = widgets.VBox([controls, ui.folder_accordion])
    ui.refresh_selector = refresh_selector
    ui.apply_selection = apply_selection
    refresh_selector()
    return ui
