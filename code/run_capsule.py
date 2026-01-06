
import os
os.environ['RUST_BACKTRACE'] = '1'
os.environ['POLARS_MAX_THREADS'] = '1'
os.environ['TOKIO_WORKER_THREADS'] = '1' 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['OMP_THREAD_LIMIT'] = '1' 
os.environ['RAYON_NUM_THREADS'] = '1'

import concurrent.futures
import contextlib
import datetime
import dateutil
import functools
import hashlib
import itertools
import json
import math
import multiprocessing
import pathlib
import uuid
import zoneinfo
from typing import Any, Iterable

import lazynwb
import polars as pl
import polars_ds as pds
import numpy as np
import pydantic_settings
import pydantic
import tqdm
import upath

import utils

ROOT_DIR = upath.UPath('s3://aind-scratch-data/dynamic-routing/psths')
decoding_parquet_path = '/root/capsule/data/all_trials_with_predict_proba.parquet'


class Params(pydantic_settings.BaseSettings):
    override_date: str | None = pydantic.Field(None, exclude=True)
    intervals_table: str = 'trials'
    align_to_col: str = 'stim_start_time'
    pre: float = 0.5
    post: float = 0.5
    default_qc_only: bool = True
    as_spike_count: bool = False
    as_binarized_array: bool = True
    bin_size_s: float = 0.001
    max_workers: int | None = pydantic.Field(None, exclude=True)
    skip_existing: bool = pydantic.Field(True, exclude=True)
    largest_to_smallest: bool = pydantic.Field(False, exclude=True)
    _start_date: datetime.date = pydantic.PrivateAttr(datetime.datetime.now(zoneinfo.ZoneInfo('US/Pacific')).date())

    def model_post_init(self, __context) -> None:        
        if self.override_date:
            self._start_date = dateutil.parser.parse(self.override_date).date()

    # --------------------------------
   
    @property
    def dir_path(self) -> upath.UPath:
        return ROOT_DIR / f"{self._start_date}"

    @pydantic.computed_field
    @property
    def datacube_version(self) -> str:
        if self.intervals_table == 'trials':
            print("Using hardcoded datacube version for trials table with grating phase info")
            return 'v0.0.274'
        ver = utils.get_datacube_dir().name.split('_')[-1]
        assert ver.startswith('v'), f"Unexpected datacube version format: {ver}"
        return ver
        
    @pydantic.computed_field
    @property
    def spike_col(self) -> str:
        if self.as_spike_count:
            return 'spike_count'
        if self.as_binarized_array:
            return 'binarized_spike_times'
        return 'spike_times'

    # set the priority of the input sources:
    @classmethod  
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        *args,
        **kwargs,
    ):
        # instantiating the class will use arguments passed directly, or provided via the command line/app panel
        # the order of the sources below defines the priority (highest to lowest):
        # - for each field in the class, the first source that contains a value will be used
        return (
            init_settings,
            pydantic_settings.sources.JsonConfigSettingsSource(settings_cls, json_file='parameters.json'),
            pydantic_settings.CliSettingsSource(settings_cls, cli_parse_args=True),
        )


def write_psths_for_area(trials: pl.DataFrame, area_label: str, params: Params, unit_ids: Iterable[str] | None = None) -> None:

    parquet_path = params.dir_path / f"{area_label}.parquet"
    if parquet_path.exists() and params.skip_existing:
        print(f'Skipping existing {parquet_path}')
        return
    print(f'\nProcessing {area_label}')
    
    area_spike_times = (
        utils.get_per_trial_spike_times(
            starts=pl.col(params.align_to_col) - params.pre,
            ends=pl.col(params.align_to_col) + params.post,
            unit_ids=unit_ids,
            trials_frame=trials,
            col_names=params.spike_col,
            as_counts=params.as_spike_count,
            as_binarized_array=params.as_binarized_array,
            bin_size_s=params.bin_size_s,
            binarized_trial_length=params.pre + params.post,
            keep_only_necessary_cols=False
        )
        .drop('bin_centers', strict=False)
        .sort('unit_id', 'trial_index')
    )
    array_len = int((params.pre + params.post) / params.bin_size_s)
    assert len(area_spike_times['binarized_spike_times'][0]) == array_len, f"Unexpected bin count in binarized_spike_times for area {area_label}"

    print(f"Writing {parquet_path}")
    (
        area_spike_times
        # sort to speed up read access:
        .sort(
            'unit_id',
            'stim_name',
            'rewarded_modality',
            'trial_index',
        )
        .with_columns(
            pl.lit(area_label).alias('area'),
        )
        .write_parquet(
            file=parquet_path.as_posix(), 
            row_group_size=100,     # reduce row groups due to size of arrays 
        )
    )
    print(f"Finished {area_label}")


if __name__ == "__main__":

    params = Params()
    print(params)

    pathlib.Path('/root/capsule/results/params.json').write_text(params.model_dump_json(indent=4))
    s3_json_path = params.dir_path.parent / f'{params.dir_path.name}.json'
    if s3_json_path.exists():
        existing_params = json.loads(s3_json_path.read_text())
        if existing_params != params.model_dump(mode='json'):
            raise ValueError(f"Params file already exists and does not match current params:\n{existing_params=}\n{params.model_dump()=}.\nDelete the data dir and params.json on S3 if you want to update parameters (or encode time in dir path)")
    else:   
        s3_json_path.write_text(params.model_dump_json(indent=4))

    nwb_files = utils.get_nwb_paths()
    # using trials with newly-added grating phase info (not in datacube asset):
    if params.intervals_table == 'trials':
        trials = (
            pl.read_parquet('s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.274/consolidated/trials.parquet')
        )
    else:
        trials = (
            lazynwb.scan_nwb(nwb_files, params.intervals_table, infer_schema_length=1).collect()
        )

    area_remapping = {
        'SCdg': 'SCm',
        'SCdw': 'SCm',
        'SCig': 'SCm',
        'SCiw': 'SCm',
        'SCop': 'SCs',
        'SCsg': 'SCs',
        'SCzo': 'SCs',
        'ECT1': 'ECT',
        'ECT2/3': 'ECT',    
        'ECT6b': 'ECT',
        'ECT5': 'ECT',
        'ECT6a': 'ECT', 
        'ECT4': 'ECT',
    }
    
    units = (
        utils.get_df('units')
        .filter(
            pl.col('default_qc') if params.default_qc_only else pl.lit(True),
        )
        .with_columns(
            pl.col('structure').replace(area_remapping)
        )
        .sort(pl.len().over('structure'), descending=params.largest_to_smallest)
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=params.max_workers or int(os.environ['CO_CPUS']), mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = []
        for (area,), df in tqdm.tqdm(units.group_by('structure', maintain_order=True), desc='Submitting processing jobs', unit='areas'):
            futures.append(executor.submit(write_psths_for_area, unit_ids=df['unit_id'], trials=trials, area_label=area, params=params))
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Processing PSTHS', unit='areas'):
            _ = future.result() # raise any errors encountered
    print(f"All finished")
    