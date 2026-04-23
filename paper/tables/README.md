# Tables Directory

This directory can be used for table source files if you prefer to keep tables separate from chapter files.

## Current Approach

Tables are currently defined directly in chapter files using the `booktabs` package. This is the recommended approach for most cases.

## Alternative: External Table Files

If you have very large tables or want to reuse tables across chapters, you can:
1. Create `.tex` files in this directory
2. Use `\input{tables/filename.tex}` in chapter files

## Table Guidelines

- Use `booktabs` package for professional formatting
- Use `siunitx` for units in table cells
- Keep tables readable (consider splitting very wide tables)
- Use consistent formatting across all tables
- Include captions and labels for cross-referencing

## Current Placeholders

The thesis includes the following table placeholders:
- Main results table (Chapter 6)
- Additional tables to be added as content is developed
