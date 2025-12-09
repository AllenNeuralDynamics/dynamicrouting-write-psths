
import os
os.environ['POLARS_MAX_THREADS'] = '1'
os.environ['TOKIO_WORKER_THREADS'] = '1' 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['OMP_THREAD_LIMIT'] = '1' 
os.environ['RAYON_NUM_THREADS'] = '1'
import contextlib
import datetime
import dateutil
import json
import concurrent.futures
import functools
import math
import multiprocessing
import pathlib
import zoneinfo
from typing import Iterable

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

class Params(pydantic_settings.BaseSettings):
    override_date: str | None = pydantic.Field(None, exclude=True)
    conv_kernel_s: float = 0.01
    correct_trials_only: bool = True
    align_to: str = 'stim_start_time'
    bin_size: float = 0.001
    pre: float = 0.5
    post: float = 0.5
    include_only_good_blocks: bool = False
    good_block_dprime_threshold: float = 1.0
    include_good_blocks_in_bad_sessions: bool = True
    min_units_across_sessions: int = pydantic.Field(500, exclude=True)
    max_workers: int | None = pydantic.Field(None, exclude=True)
    n_null_iterations: int = 100

    # --------------------------------
    @property
    def block_label(self) -> str:
        if self.include_only_good_blocks and self.include_good_blocks_in_bad_sessions:
            return "good-blocks_all-sessions"
        elif self.include_only_good_blocks:
            return "good-blocks_good-sessions"
        else:
            return "all-blocks_good-sessions"

    @property
    def dir_path(self) -> upath.UPath:
        if self.override_date:
            dt = dateutil.parser.parse(self.override_date)
        else:
            dt = datetime.datetime.now(zoneinfo.ZoneInfo('US/Pacific'))
        return ROOT_DIR / f"{dt.date()}_{self.conv_kernel_s*1000:.0f}ms_{self.block_label}"

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


def psth(
    df: pl.DataFrame,
    response_col: str,
    duration_col: str | pl.Expr,
    group_by: str | list[str] = "index",
    bin_size=0.001,
    conv_kernel=0.005,
    parallel=True,
    with_original=False,
) -> pl.DataFrame:
    """
    Compute the peristimulus time histogram (PSTH) for a given response and duration.
    """
    if isinstance(group_by, str) or not isinstance(group_by, Iterable):
        group_by = [group_by]
    if isinstance(duration_col, str):
        duration = pl.col(duration_col)
    n_bins = duration.first().truediv(bin_size).ceil().cast(int)
    bin_edges = pl.linear_space(
        0, duration.first(), 1 + n_bins, closed="both"
    )
    bin_centers = pl.linear_space(
        bin_size / 2, duration.first() - bin_size / 2, n_bins
    ).alias("bin_centers")
    conv_kernel_size: int = math.ceil(conv_kernel / bin_size)
    if with_original:
        extra_cols = [
            bin_centers,
            pl.col(response_col)
            .hist(bins=bin_edges, include_breakpoint=False)
            .alias("unconv_psth"),
        ]
    else:
        extra_cols = [bin_centers]
    updated_df = (
        df.with_row_index()
        .explode(response_col)
        .group_by(group_by)
        .agg(
            pl.all().exclude(response_col),
            psth=pds.convolve(
                pl.col(response_col).hist(bins=bin_edges, include_breakpoint=False),
                kernel=[1/bin_size] * conv_kernel_size,
                mode="same",
                parallel=parallel,
            ).truediv(
                conv_kernel_size * pl.col("index").n_unique()
            ),  # ,
            # psth=pds.convolve(pl.col(response_col).hist(bins=bin_edges, include_breakpoint=False), kernel=[1]*conv_kernel_size, mode='full', parallel=parallel).slice(conv_kernel_size-1, n_bins).truediv(conv_kernel_size * pl.col('index').n_unique()),#,
            *extra_cols,
        )
        ## DEBUGGING: check lengths of list columns created above -  should all be equal
        # .with_columns(
        #     pl.col('psth', 'unconv_psth', 'bin_centers').list.len(),
        # )
        .drop("index")
    )
    assert len(updated_df) == len(df.group_by(group_by).agg(pl.all().first())), f"After adding PSTH column, dataframe changed length: {len(updated_df)=}, len(original_df)={len(df)}"
    return updated_df


def write_psths_for_area(unit_ids: Iterable[str], trials: pl.DataFrame, area: str, params: Params, skip_existing: bool = True) -> None:

    parquet_path = params.dir_path / f'{area}.parquet'
    if skip_existing and parquet_path.exists():
        print(f'Skipping {area}: parquet already on S3')
        return None
    
    area_spike_times = utils.get_per_trial_spike_times(
        starts=pl.col(params.align_to) - params.pre,
        ends=pl.col(params.align_to) + params.post,
        unit_ids=unit_ids,
        trials_frame=(
            trials
            .filter(
                pl.col('is_correct') if params.correct_trials_only else pl.lit(True),
                ~pl.col('is_instruction'),
                pl.col('is_aud_target') | pl.col('is_vis_target'),
            )
            .with_columns(session_id=pl.col('_nwb_path').str.split('/').list.get(-1).str.strip_suffix('.nwb'))
        ),
        as_counts=False,
        as_binarized_array=False,
        binarized_trial_length=1.0,
        keep_only_necessary_cols=False
    ).sort('unit_id', 'trial_index')

    contexts = (
            pl.col('is_aud_target') & pl.col('is_aud_rewarded'),
            pl.col('is_aud_target') & ~pl.col('is_aud_rewarded'),
            pl.col('is_vis_target') & pl.col('is_vis_rewarded'),
            pl.col('is_vis_target') & ~pl.col('is_vis_rewarded')
        )

    sessions_with_all_contexts = functools.reduce(np.intersect1d, [area_spike_times.filter(context)['session_id'].unique() for context in contexts])

    context_state_to_context_string = {
        4: 'AA', #auditory target, auditory context
        3: 'AV', #auditory target, visual context
        2: 'VA', #visual target, auditory context
        1: 'VV'  #visual target, visual context
    }

    area_spike_times = (
        area_spike_times
        .filter( 
            pl.col('session_id').is_in(sessions_with_all_contexts)
        )
        .with_columns(
            (pl.col('is_aud_target').cast(pl.Int8) * 2 + pl.col('is_aud_rewarded').cast(pl.Int8) + 1).cast(pl.Utf8).replace(context_state_to_context_string).alias('context_state'),
        )
        .drop('bin_centers', strict=False)
    )

    unit_psths = (
        area_spike_times
        .with_columns(
            pl.lit(1).alias('duration'), # needed for psth
        )
        .pipe(psth, response_col='n_spikes', duration_col='duration', group_by=['session_id', 'context_state', 'unit_id'], conv_kernel=params.conv_kernel_s, bin_size=params.bin_size,)
        .select('session_id', 'unit_id', 'context_state', 'psth')
        .with_columns(
            pl.lit(None).cast(pl.Int32).alias('null_iteration'),
        )
    )
    extra_dfs = []
    if params.n_null_iterations:
        for i in tqdm.tqdm(range(params.n_null_iterations), total=params.n_null_iterations, unit='iterations', desc=f'Computing null PSTHs for {area}'):

            null_unit_psths = (
                area_spike_times
                .with_columns(
                    pl.lit(1).alias('duration'),
                )
                #shuffle context labels within session groups
                .group_by('session_id', 'unit_id', pl.col('context_state').str.head(1).alias('stim')).agg(
                    pl.all())
                .with_columns(
                    pl.col('context_state').list.sample(n=pl.col('context_state').list.len(), shuffle=True, with_replacement=False, seed=i),)
                .explode(pl.all().exclude('session_id', 'unit_id', 'stim'))
                .select(
                    'n_spikes', 'duration', 'session_id', 'unit_id', 'context_state',
                )
                
                #make psths
                .pipe(psth, response_col='n_spikes', duration_col='duration', group_by=['session_id', 'context_state', 'unit_id'], conv_kernel=params.conv_kernel_s, bin_size=params.bin_size)
                
                .select('session_id', 'unit_id', 'context_state', 'psth')
                .with_columns(
                    pl.lit(i).alias('null_iteration'),
                )
            )
            extra_dfs.append(null_unit_psths)

    if extra_dfs:
        unit_psths = pl.concat([unit_psths] + extra_dfs)

    unit_psths.write_parquet(parquet_path.as_posix())
    print(f"Wrote {parquet_path.as_posix()}")


if __name__ == "__main__":

    params = Params()

    pathlib.Path('/root/capsule/results/params.json').write_text(params.model_dump_json(indent=4))
    s3_json_path = params.dir_path.parent / f'{params.dir_path.name}.json'
    if s3_json_path.exists():
        existing_params = json.loads(s3_json_path.read_text())
        if existing_params != params.model_dump():
            raise ValueError(f"Params file already exists and does not match current params:\n{existing_params=}\n{params.model_dump()=}.\nDelete the data dir and params.json on S3 if you want to update parameters (or encode time in dir path)")
    else:   
        s3_json_path.write_text(params.model_dump_json(indent=4))

    nwb_dir_path = pathlib.Path('/root/capsule/data/dynamicrouting_datacube_v0.0.272/nwb')
    nwb_files = list(nwb_dir_path.glob('*.nwb'))
    sessions = lazynwb.scan_nwb(nwb_files, 'session').collect()
    performance = lazynwb.scan_nwb(nwb_files, 'intervals/performance').collect()
    session_table = pl.read_parquet('/root/capsule/data/dynamicrouting_datacube_v0.0.272/session_table.parquet')
    good_behavior_sessions = session_table.filter(pl.col('is_good_behavior'))['session_id'].to_list()
    sessions_to_analyze = (
        sessions
        .filter(pl.col('keywords').list.contains('production'),
            ~pl.col('keywords').list.contains('templeton'),
            ~pl.col('keywords').list.contains('injection_perturbation'),
            ~pl.col('keywords').list.contains('injection_control'),
            ~pl.col('keywords').list.contains('opto_perturbation'),
            ~pl.col('keywords').list.contains('opto_control'),
            ~pl.col('keywords').list.contains('issues'),
            ~pl.col('keywords').list.contains('naive'),
            ~pl.col('keywords').list.contains('context_naive'),
            pl.lit(True) if (params.include_good_blocks_in_bad_sessions and params.include_only_good_blocks) else pl.col('session_id').is_in(good_behavior_sessions),
        )
    )
    trials = (
        lazynwb.scan_nwb(sessions_to_analyze['_nwb_path'], 'trials').collect()
    )
    if params.include_only_good_blocks:
        trials = (
            trials
            .join(
                performance.filter(
                    pl.col('cross_modality_dprime') >= params.good_block_dprime_threshold,
                    pl.col('n_contingent_rewards') >= 10,
                ),
                on=['_nwb_path', 'block_index'], 
                how='semi',
            )
        )

    units = lazynwb.scan_nwb(nwb_files, 'units', infer_schema_length=1)

    area_remapping = {
        'SCdg': 'SCm',
        'SCdw': 'SCm',
        'SCig': 'SCm',
        'SCiw': 'SCm',
        'SCop': 'SCs',
        'SCsg': 'SCs',
        'SCzo': 'SCs',
    }

    well_sampled_good_units = (
        units
        .filter(
            pl.col('default_qc'),
            pl.col('_nwb_path').is_in(sessions_to_analyze['_nwb_path'].implode()),
        )
        .select(
            # 'spike_times',
            '_nwb_path',
            'unit_id',
            'structure',
        )
        .with_columns(
            pl.col('structure').replace(area_remapping)
        )

        # apply threshold on n_units:
        .drop_nulls('structure')
        .filter((pl.n_unique('unit_id')>=params.min_units_across_sessions).over('structure')) # checks total across all sessions
        .collect()
        .sort(pl.len().over('structure'))
    )
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=params.max_workers or int(os.environ['CO_CPUS']), mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = []
        for (area,), df in tqdm.tqdm(well_sampled_good_units.group_by('structure', maintain_order=True), desc='Submitting processing jobs', unit='areas'):
            futures.append(executor.submit(write_psths_for_area, unit_ids=df['unit_id'], trials=trials, area=area, params=params))
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Processing PSTHS', unit='areas'):
            _ = future.result() # raise any errors encountered
    print(f"All finished")