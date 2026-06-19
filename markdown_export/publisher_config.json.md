# `publisher_config.json`

Source path: `Confluence_Publisher/publisher_config.json`

```json
{
    "project_id": "",
    "dataset_id": "",

    "target_service_account": null,
    "quota_project_id":       null,
    "token_lifetime":         3600,

    "output_gcs_uri":  "gs://<your-bucket>/grid_search_ml/",

    "confluence_base_url":           "https://<your-confluence-host>",
    "confluence_space_key":          "",
    "confluence_parent_page_id":     "",
    "confluence_flavor":             "server",
    "confluence_verify_ssl":         false,
    "confluence_auth_connection_id": "confluence_default",
    "confluence_row_cap":            5000,
    "confluence_lookback_days":      2,
    "confluence_wide_cell_columns":  ["rca_top_error", "rca_stack_hint", "latest_message"]
}
```
