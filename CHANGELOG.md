# Changelog

## 1.10.3 - 2026-10-27

### Added (source groups)
- `CatchAllClient.list_source_groups(page=1, page_size=100)` and the async equivalent, wired to
  `GET /catchAll/source-groups` (returns `source_groups` with slug, name, description, plus
  `total`, `page`, `page_size`; `page_size` max 500).
- `source_groups` (list of slugs, max 20) on `submit_job` and `search` for both
  `CatchAllClient` and `AsyncCatchAllClient`. More than 20 slugs raises `ValueError`.
- `CatchAllTools`: new tool `catchall_list_source_groups` (params `page`, `page_size`) and an
  optional `source_groups` argument on `catchall_search_data` (more than 20 returns an error string).
- README section "Source Groups"; tests added to `tests/test_client.py` and
  `tests/unit_tests/test_catchall_tools_unit.py`.
