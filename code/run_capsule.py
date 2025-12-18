
import itertools
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
    conv_kernel_s: float = 0.01
    align_to: str = 'stim_start_time'
    decoder_areas_to_average: list[str] = pydantic.Field(default_factory=lambda: sorted(['ACAd', 'AId', 'AIp', 'FRP', 'ILA', 'MOs', 'MOp', 'ORBl', 'ORBvl', 'PL', 'SSp', 'SSs', 'MRN', 'SCm', 'CP']))
    bin_size: float = 0.001
    pre: float = 0.5
    post: float = 0.5
    include_only_good_blocks: bool = True
    good_block_dprime_threshold: float = 1.0
    include_good_blocks_in_bad_sessions: bool = False
    min_units_across_sessions: int = pydantic.Field(500, exclude=True)
    max_workers: int | None = pydantic.Field(None, exclude=True)
    n_null_iterations: int = 100
    skip_existing: bool = pydantic.Field(True, exclude=True)
    largest_to_smallest: bool = pydantic.Field(False, exclude=True)
    _start_date: datetime.date = pydantic.PrivateAttr(datetime.datetime.now(zoneinfo.ZoneInfo('US/Pacific')).date())

    def model_post_init(self, __context) -> None:        
        if self.override_date:
            self._start_date = dateutil.parser.parse(self.override_date).date()

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
        return ROOT_DIR / f"{self._start_date}_{self.conv_kernel_s*1000:.0f}ms_{self.block_label}"

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


def write_psths_for_area(unit_ids: Iterable[str], trials: pl.DataFrame, area: str, params: Params) -> None:

    all_conditions = (
        # each group below has the same stim
        # multiple nulls are created for each group based on pairs of expressions within the group
            (
                ['is_aud_target', 'is_aud_rewarded', 'is_hit'], # hit aud
                ['is_aud_target', 'is_aud_rewarded', 'is_miss'], # miss aud
                ['is_aud_target', 'is_vis_rewarded', 'is_false_alarm'], # FA aud
                ['is_aud_target', 'is_vis_rewarded', 'is_correct_reject'], # CR aud
                ['is_aud_target', 'is_vis_rewarded', 'is_false_alarm', 'is_decoder_correct', 'is_decoder_confident'], # FA aud for confident correct decoder
                ['is_aud_target', 'is_vis_rewarded', 'is_correct_reject', 'is_decoder_correct'], #
                ['is_aud_target', 'is_vis_rewarded', 'is_correct_reject', 'is_decoder_incorrect'], #
            ),

            # vis targets:
            (
                ['is_vis_target', 'is_vis_rewarded', 'is_hit'], # hit vis
                ['is_vis_target', 'is_vis_rewarded', 'is_miss'], # miss vis
                ['is_vis_target', 'is_aud_rewarded', 'is_false_alarm'], # FA vis
                ['is_vis_target', 'is_aud_rewarded', 'is_correct_reject'], # CR vis
                ['is_vis_target', 'is_aud_rewarded', 'is_false_alarm', 'is_decoder_correct', 'is_decoder_confident'], # FA vis for confident correct decoder
                ['is_vis_target', 'is_aud_rewarded', 'is_correct_reject', 'is_decoder_correct'], # 
                ['is_vis_target', 'is_aud_rewarded', 'is_correct_reject', 'is_decoder_incorrect'], # 
            ),
            (
                ['is_vis_target', 'is_grating_phase_half', 'is_vis_rewarded', 'is_hit'], # hit vis
                ['is_vis_target', 'is_grating_phase_half', 'is_vis_rewarded', 'is_miss'], # miss vis
                ['is_vis_target', 'is_grating_phase_half', 'is_aud_rewarded', 'is_false_alarm'], # FA vis
                ['is_vis_target', 'is_grating_phase_half', 'is_aud_rewarded', 'is_correct_reject'], # CR vis
                ['is_vis_target', 'is_grating_phase_half', 'is_aud_rewarded', 'is_false_alarm', 'is_decoder_correct', 'is_decoder_confident'], # FA vis for confident correct decoder
                ['is_vis_target', 'is_grating_phase_half', 'is_aud_rewarded', 'is_correct_reject', 'is_decoder_correct'], # 
                ['is_vis_target', 'is_grating_phase_half', 'is_aud_rewarded', 'is_correct_reject', 'is_decoder_incorrect'], # 
            ),
            (
                ['is_vis_target', 'is_grating_phase_zero', 'is_vis_rewarded', 'is_hit'], # hit vis
                ['is_vis_target', 'is_grating_phase_zero', 'is_vis_rewarded', 'is_miss'], # miss vis
                ['is_vis_target', 'is_grating_phase_zero', 'is_aud_rewarded', 'is_false_alarm'], # FA vis
                ['is_vis_target', 'is_grating_phase_zero', 'is_aud_rewarded', 'is_correct_reject'], # CR vis
                ['is_vis_target', 'is_grating_phase_zero', 'is_aud_rewarded', 'is_false_alarm', 'is_decoder_correct', 'is_decoder_confident'], # FA vis for confident correct decoder
                ['is_vis_target', 'is_grating_phase_zero', 'is_aud_rewarded', 'is_correct_reject', 'is_decoder_correct'], # 
                ['is_vis_target', 'is_grating_phase_zero', 'is_aud_rewarded', 'is_correct_reject', 'is_decoder_incorrect'], # 
            ),

            # aud nontargets:
            (
                ['is_aud_nontarget', 'is_aud_rewarded', 'is_false_alarm'], # FA aud nontarget aud context
                ['is_aud_nontarget', 'is_vis_rewarded', 'is_false_alarm'], # FA aud nontarget vis context
                ['is_aud_nontarget', 'is_aud_rewarded', 'is_correct_reject'], # CR aud nontarget aud context
                ['is_aud_nontarget', 'is_vis_rewarded', 'is_correct_reject'], # CR aud nontarget vis context
            ),

            # vis nontargets:
            (
                ['is_vis_nontarget', 'is_grating_phase_zero', 'is_aud_rewarded', 'is_false_alarm'], # FA vis nontarget aud context
                ['is_vis_nontarget', 'is_grating_phase_zero', 'is_vis_rewarded', 'is_false_alarm'], # FA vis nontarget vis context
                ['is_vis_nontarget', 'is_grating_phase_zero', 'is_aud_rewarded', 'is_correct_reject'], # CR vis nontarget aud context
                ['is_vis_nontarget', 'is_grating_phase_zero', 'is_vis_rewarded', 'is_correct_reject'], # CR vis nontarget vis context
            ),
            (
                ['is_vis_nontarget', 'is_grating_phase_half', 'is_aud_rewarded', 'is_false_alarm'], # FA vis nontarget aud context
                ['is_vis_nontarget', 'is_grating_phase_half', 'is_vis_rewarded', 'is_false_alarm'], # FA vis nontarget vis context
                ['is_vis_nontarget', 'is_grating_phase_half', 'is_aud_rewarded', 'is_correct_reject'], # CR vis nontarget aud context
                ['is_vis_nontarget', 'is_grating_phase_half', 'is_vis_rewarded', 'is_correct_reject'], # CR vis nontarget vis context
            ),
        )
    
    condition_cols = set()
    for conds in all_conditions:
        for cond in conds:
            condition_cols.update(cond)
    condition_cols = sorted(condition_cols)

    null_condition_pairs: list[list[tuple[list[str], list[str]]]] = []
    for condition_group in all_conditions:
        null_condition_pairs.append(list(itertools.combinations(condition_group, 2)))
    
    print(f'\nProcessing {area}')
    
    area_spike_times = (
        utils.get_per_trial_spike_times(
            starts=pl.col(params.align_to) - params.pre,
            ends=pl.col(params.align_to) + params.post,
            unit_ids=unit_ids,
            trials_frame=(
                trials
                .filter(~pl.col('is_instruction'))
            ),
            as_counts=False,
            as_binarized_array=False,
            binarized_trial_length=1.0,
            keep_only_necessary_cols=False
        )
        .drop('bin_centers', strict=False)
        .sort('unit_id', 'trial_index')
    )

    def write(df, path: upath.UPath) -> None:
        (
            df
            .with_columns(
                pl.lit(area).alias('area'),
            )
            .write_parquet(path.as_posix())
        )

    def get_parquet_path(to_hash: Any) -> upath.UPath:
        return params.dir_path / area / f"{area}_{hash(str(to_hash))}.parquet"

    for stim_idx, conditions in enumerate(all_conditions):
        for condition in conditions:
            if (path := get_parquet_path(condition)).exists():
                continue
            unit_psths = (
                area_spike_times
                .filter(*condition)
                .with_columns(
                    pl.lit(1).alias('duration'), # needed for psth
                )
                .pipe(psth, response_col='n_spikes', duration_col='duration', group_by=['session_id', 'unit_id', *condition_cols, 'predict_proba'], conv_kernel=params.conv_kernel_s, bin_size=params.bin_size,)
                .select('session_id', 'unit_id', *condition_cols, 'psth', 'predict_proba')
                .with_columns(
                    pl.lit(condition).alias('condition_filter'),
                    pl.lit(None).alias('null_iteration'),
                    pl.lit(None).alias('null_condition_1_filter'),
                    pl.lit(None).alias('null_condition_2_filter'),
                    pl.lit(None).alias('null_condition_index')
                )
            )
            write(unit_psths, path)

        if params.n_null_iterations:
            for null_condition_pair in null_condition_pairs[stim_idx]:
                
                if (path := get_parquet_path(null_condition_pair)).exists():
                    continue

                null_dfs = []
                
                for i in range(params.n_null_iterations):
                    null_unit_psths = (
                        area_spike_times
                        .with_columns(
                            pl.lit(1).alias('duration'),
                        )
                        .with_columns(
                            pl.when(*null_condition_pair[0]).then(pl.lit(1)).when(*null_condition_pair[1]).then(pl.lit(2)).alias('null_condition_index')
                        )
                        .drop_nulls('null_condition_index')
                        .filter(
                            pl.col('null_condition_index').n_unique().eq(2).over('session_id')
                        )
                        #shuffle context labels within session groups
                        .group_by('session_id', 'unit_id', *condition_cols)
                        .agg(pl.all())
                        .with_columns(
                            pl.col('null_condition_index').list.sample(fraction=1, shuffle=True, with_replacement=False, seed=i),
                        )
                        .explode(pl.all().exclude('session_id', 'unit_id', *condition_cols))
                        
                        #make psths
                        .pipe(psth, response_col='n_spikes', duration_col='duration', group_by=['session_id', 'unit_id', *condition_cols, 'null_condition_index', 'predict_proba'], conv_kernel=params.conv_kernel_s, bin_size=params.bin_size)
                                                    .select('session_id', 'unit_id', *condition_cols, 'predict_proba', 'psth', 'null_condition_index')
                        .with_columns(
                            pl.lit(None).cast(pl.List(str)).alias('condition_filter'),
                            pl.lit(i).alias('null_iteration'),
                            pl.lit(null_condition_pair[0]).alias('null_condition_1_filter'),
                            pl.lit(null_condition_pair[1]).alias('null_condition_2_filter'),
                        )
                    )
                    null_dfs.append(null_unit_psths)
                write(pl.concat(null_dfs, how="diagonal_relaxed"), path)

    print(f"Finished {area}")


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
        pl.read_parquet('s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.274/consolidated/trials.parquet')
    )
    if params.include_only_good_blocks:
        trials = (
            trials
            .join(
                (
                    performance
                    .with_columns(pl.col('_nwb_path').str.split('/').list.get(-1).str.strip_suffix('.nwb').alias('session_id'))
                    .filter(
                        pl.col('cross_modality_dprime') >= params.good_block_dprime_threshold,
                        pl.col('n_contingent_rewards') >= 10,
                    )
                ),
                on=['session_id', 'block_index'], 
                how='semi',
            )
        )
    
    cols = [f"{a}_predict_proba" for a in params.decoder_areas_to_average]

    decoding_df = (
        pl.read_parquet(decoding_parquet_path)
        .with_columns(pl.mean_horizontal(cols).alias('predict_proba'))
        .with_columns(
            predict_proba_quintile=pl.col('predict_proba').cut([0.2, 0.4, 0.6, 0.8], include_breaks=False)
        )
        .select('session_id', 'trial_index', 'predict_proba', 'predict_proba_quintile')
    )
    trials = (
        trials
        .join(decoding_df, on=['trial_index', 'session_id'], how='inner')
        .with_columns(
            is_decoder_confident=pl.col('predict_proba').sub(0.5).abs().gt(0.1),
            is_decoder_correct = ((pl.col('predict_proba')<0.5)&(pl.col('is_aud_rewarded'))) | ((pl.col('predict_proba')>0.5)&(pl.col('is_vis_rewarded'))),
            is_grating_phase_zero=pl.col('grating_phase').eq(0),
            is_grating_phase_half=pl.col('grating_phase').eq(0.5),
        ) 
        .with_columns(
            is_decoder_incorrect=~pl.col('is_decoder_correct'),
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
        .sort(pl.len().over('structure'), descending=params.largest_to_smallest)
    )
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=params.max_workers or int(os.environ['CO_CPUS']), mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = []
        for (area,), df in tqdm.tqdm(well_sampled_good_units.group_by('structure', maintain_order=True), desc='Submitting processing jobs', unit='areas'):
            futures.append(executor.submit(write_psths_for_area, unit_ids=df['unit_id'], trials=trials, area=area, params=params))
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Processing PSTHS', unit='areas'):
            _ = future.result() # raise any errors encountered
    print(f"All finished")
    