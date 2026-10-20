"""Custom Streamlit components (plain HTML/JS folders, no build step). Declared here, in an importable module, because
declare_component cannot find the module of a page script."""

from pathlib import Path

import streamlit.components.v1 as components

# the 2.4 marking editor: args photo, photo_id, file_name, batch_id, batch_start, sample_id, date, record;
# returns {'photo_id', 'record', 'reviewed', 'saved_at'} after Save, else the default
check_image = components.declare_component("check_image", path=str(Path(__file__).parent / "check_image"))
