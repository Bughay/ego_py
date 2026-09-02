---
name: file_archivist
description: "Save text artifacts into workspace files with clear names, then verify them"
---
To archive an artifact into the workspace:

1. Choose a descriptive lowercase filename (words separated by underscores) with the right extension (.txt for prose, .json for structured data).
2. Call the `write_file` tool with the filename and the full artifact text.
3. Verify the save by calling `read_file` on the same path.
4. If a file with that name already exists, pick a new name instead of overwriting.
