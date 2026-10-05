"""Author timeF historical metadata only; never a fitted/business channel."""
MODELS = ('iTransformer', 'TimeMixer', 'TimeXer')
MODE = 'native_time_mark_v1'
FEATURES = ('HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear')


def policy(p):
    if p['model'] not in MODELS:
        return None
    freq = p['structure']['freq']
    minute=freq=='t'and p['dataset']in ('ETTm1','ETTm2')
    if freq != 'h'and not minute:
        raise ValueError('native timeF supports frozen hourly or source-admitted ETT minute interface')
    value=dict(mode=MODE, freq=freq, features=list(('MinuteOfHour',)+FEATURES if minute else FEATURES), K=5 if minute else 4,
                source='bound timestamp column, exact historical x window',
                fitted=False, future=False, business_channels_added=0)
    if minute:value['sampling_interval']='15min'
    return value


def marks(times, freq='h'):
    import numpy as np
    import pandas as pd
    if freq not in ('h','t'):
        raise ValueError('frozen native mark frequency changed')
    index = pd.DatetimeIndex(times)
    if index.hasnans or index.tz is not None:
        raise ValueError('invalid or timezone-bearing bound timestamp')
    value=np.stack((index.hour / 23.0 - .5, index.dayofweek / 6.0 - .5,
                     (index.day - 1) / 30.0 - .5,
                     (index.dayofyear - 1) / 365.0 - .5), axis=1).astype('float32')
    return np.column_stack((index.minute / 59.0 - .5,value)).astype('float32') if freq=='t' else value


def batch_parts(batch):
    if len(batch) not in (2, 3):
        raise ValueError('exact x/y/[historical mark] batch')
    return batch[0], batch[1], batch[2] if len(batch) == 3 else None


def synthetic(p, size):
    """No RNG draw; the same historical clock construction in reference/parallel."""
    import pandas as pd
    import numpy as np
    import torch
    if not p.get('time_mark'):
        return None
    value = marks(pd.date_range('2022-09-01', periods=p['T'], freq=p['time_mark'].get('sampling_interval',p['time_mark']['freq'])),p['time_mark']['freq'])
    return torch.from_numpy(np.broadcast_to(value, (size, *value.shape)).copy())


def metadata(p):
    return dict(time_mark=p['time_mark'], time_mark_shape=[p['T'], p['time_mark']['K']]) if p.get('time_mark') else {}
