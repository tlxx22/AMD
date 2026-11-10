"""CSV/JSON and figures share the same values; never invoke a model."""
import csv
from pathlib import Path
import numpy as np
from .contract import read, write_new


def table(path, rows):
    if not rows:
        raise ValueError('empty table')
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure(path, x, series, xlabel, ylabel):
    """Same geometry to vector SVG and Pillow PNG; no new plotting dependency."""
    points = {name: list(zip(range(len(x)), map(float, y))) for name, y in series.items()}
    chart(path, points, xlabel, ylabel, tick_labels=list(map(str, x)))


def chart(path, points, xlabel, ylabel, tick_labels=None, scatter=False):
    from html import escape
    from PIL import Image, ImageDraw, ImageFont
    width, height = 1400, 800
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    def text(x, y, value, color='#222222'):
        draw.text((x, y), value, font=font, fill=color)
        svg.append(f'<text x="{x}" y="{y+18}" fill="{color}" font-family="sans-serif" font-size="18">{escape(value)}</text>')
    def line(coords, color, stroke=2):
        draw.line(coords, fill=color, width=stroke)
        value = ' '.join(f'{x:.3f},{y:.3f}' for x, y in coords)
        svg.append(f'<polyline points="{value}" fill="none" stroke="{color}" stroke-width="{stroke}"/>')
    values = [xy for data in points.values() for xy in data]
    if not values or not np.isfinite(values).all():
        raise ValueError('finite plot values required')
    xmin, xmax = min(v[0] for v in values), max(v[0] for v in values)
    ymin, ymax = min(v[1] for v in values), max(v[1] for v in values)
    if xmax == xmin:
        xmax = xmin+1
    gap = (ymax-ymin)*.08 if ymax != ymin else max(abs(ymin)*.08, .01)
    ymin, ymax = ymin-gap, ymax+gap
    def pixel(x, y):
        return 115+(x-xmin)/(xmax-xmin)*1170, 635-(y-ymin)/(ymax-ymin)*470
    for i in range(6):
        y = ymin+(ymax-ymin)*i/5
        yy = pixel(xmin, y)[1]
        line([(115, yy), (1285, yy)], '#dddddd', 1)
        text(8, yy-10, f'{y:.5g}')
    for value in np.linspace(xmin, xmax, min(5, len(tick_labels)) if tick_labels else 5):
        label = tick_labels[round(value)] if tick_labels else f'{value:.5g}'
        text(pixel(value, ymin)[0]-40, 650, label)
    palette = ['#0068ad', '#dc5310', '#149173', '#9a4caa', '#bf3759', '#7c6c16', '#2b899d', '#505050']
    for i, (name, data) in enumerate(points.items()):
        color = palette[i % len(palette)]
        coordinates = [pixel(x, y) for x, y in data]
        if not scatter and len(coordinates) > 1:
            line(coordinates, color, 3)
        for x, y in coordinates:
            draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
            svg.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3" fill="{color}"/>')
        text(115+(i % 4)*300, 80+(i//4)*30, name, color)
    text(115, 25, ylabel)
    text(400, 725, xlabel)
    svg.append('</svg>')
    png, vector = Path(str(path)+'.png'), Path(str(path)+'.svg')
    if png.exists() or vector.exists():
        raise FileExistsError(path)
    vector.write_text('\n'.join(svg)+'\n')
    image.save(png)


def completed(config, folder):
    folder = Path(folder)
    out = folder/'plots'
    out.mkdir(exist_ok=False)
    e, sensitivity = [], []
    for job in config['jobs']:
        if job['phase'] in ('E', 'S'):
            r = read(folder/job['id']/'result.json')
            if job['phase'] == 'E':
                e.append(dict(dataset=job['dataset'], model=job['model'], batch=job['batch'], median_batch_ms=r['median_batch_ms'], median_sample_ms=r['median_sample_ms'], stdev_batch_ms=r['stdev_batch_ms'], registered_parameters=r['parameters'], peak_allocated_bytes=r['peak']['max_allocated'], resident_allocated_bytes=r['resident']['allocated'], peak_reserved_bytes=r['peak']['max_reserved']))
            else:
                sensitivity.append(dict(dataset=job['dataset'], G=job['G'], R=job['R'], mse=r['validation']['mse'], mae=r['validation']['mae'], best_epoch=r['best_epoch'], epochs=r['epochs'], updates=r['updates'], provenance='fresh validation only'))
    table(out/'efficiency.csv', e)
    table(out/'sensitivity-fresh.csv', sensitivity)
    for dataset in ('UrbanEV', 'PJM'):
        for batch in (1, 16):
            for metric in ('mse', 'mae'):
                points = {}
                for row in [r for r in e if r['dataset'] == dataset and r['batch'] == batch]:
                    score = next(r for r in config['main_results'] if r['dataset'] == dataset and r['model'] == row['model'] and r['H'] == (6 if dataset == 'UrbanEV' else 24))
                    points[row['model']] = [(row['median_batch_ms'], score[metric])]
                chart(out/(dataset+f'-error-latency-b{batch}-'+metric), points, 'Median batch latency (ms)', 'Original saved test '+metric.upper(), scatter=True)
        center = config['sensitivity_centers'][dataset]
        rows = [r for r in sensitivity if r['dataset'] == dataset]
        rows.append(dict(dataset=dataset, G=5, R=4, mse=center['validation']['mse'], mae=center['validation']['mae'], best_epoch=center['best_epoch'], epochs=center['epochs'], updates=center['updates'], provenance='original center reference; zero new training'))
        table(out/(dataset+'-sensitivity.csv'), rows)
        for metric in ('mse', 'mae'):
            for axis, fixed in [('G', 'R'), ('R', 'G')]:
                chosen = sorted([r for r in rows if r[fixed] == (4 if fixed == 'R' else 5)], key=lambda r: r[axis])
                figure(out/(dataset+'-sensitivity-'+axis+'-'+metric), [r[axis] for r in chosen], {'UV': [r[metric] for r in chosen]}, axis, 'validation '+metric.upper())
    for metric in ('mse', 'mae'):
        series = {}
        for model in ('ModernTCN', 'TimeMixer', 'UV'):
            r = read(folder/('P-'+model)/'result.json')
            series[model] = r['metrics'][metric]
        figure(out/('PJM-position-'+metric), range(1, 25), series, 'Position inside fixed P=24', 'validation '+metric.upper())
        table(out/('PJM-position-'+metric+'.csv'), [dict(position=i+1, **{m: v[i] for m, v in series.items()}) for i in range(24)])
    selection = read(folder/'case-selection.json')
    case = {m: np.load(folder/('C-'+m)/'predictions.npz') for m in ('ModernTCN', 'TimeMixer', 'UV')}
    for region in range(2):
        sl = slice(region*168, (region+1)*168)
        truth = case['UV']['physical_truth'][sl, 0, 0]
        if any(not np.array_equal(v['truth'], case['UV']['truth']) for v in case.values()):
            raise ValueError('case truth/order mismatch')
        series = {'Truth': truth, **{m: v['physical_prediction'][sl, 0, 0] for m, v in case.items()}}
        figure(out/('UrbanEV-node-'+str(selection['nodes'][region])), selection['target_times'], series, 'Target time (H6 rolling P1)', 'Volume (original dataset units)')
        table(out/('case-'+str(region)+'.csv'), [dict(target_time=selection['target_times'][i], **{m: float(v[i]) for m, v in series.items()}) for i in range(168)])
        write_new(out/('case-'+str(region)+'-metrics.json'), {m: dict(mse=float(np.square(v-truth).mean()), mae=float(np.abs(v-truth).mean())) for m, v in series.items() if m != 'Truth'})
    return dict(status='completed', test=0, note='sensitivity and measurements do not replace main-table results')
