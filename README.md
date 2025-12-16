
resulting dataframes (1 per area) have the following schema (one PSTH per row):

```
{
    'session_id': String,
    'unit_id': String,
    'is_aud_target': Boolean,
    'is_vis_target': Boolean,
    'is_aud_nontarget': Boolean,
    'is_vis_nontarget': Boolean,
    'is_aud_rewarded': Boolean,
    'is_vis_rewarded': Boolean,
    'is_response': Boolean,
    'is_hit': Boolean,
    'is_miss': Boolean,
    'is_correct_reject': Boolean,
    'is_false_alarm': Boolean,
    'psth': List(Float64),
    'predict_proba': Categorical,               # indicates range, eg "(0.2, 0.4]" or "(0.8, inf]"
    'null_condition': Int32,                    # denotes which of the pair of conditions the row corresponds to: either 1 or 2 (or Null for regular PSTHs)
    'null_iteration': Int32,                    # 0-99
    'null_pair_id': Int32,                      # a unique integer for each pair of conditions (ie for each id there will be rows with null_condition==1 and null_condition==2)
    'null_condition_1_filter': List(String),    # list of column names that will all be True for this PSTH, eg ["is_aud_target", "is_aud_rewarded", "is_hit"]
    'null_condition_2_filter': List(String),
    'area': String
 }
```

TODO
- break up null tables?
- add area acronym
