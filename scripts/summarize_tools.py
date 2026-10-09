"""Score saved structured predictions; hidden procedure labels enter scoring only."""

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    from newsvendor.io import digest, jsonl, lines, read, require, write
    from newsvendor.suite import canonical

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['dev', 'test'], required=True)
    args = parser.parse_args()
    saved = lines(args.predictions)
    ids = {r.get('id', r.get('key')) for r in saved}
    rows = {r['id']: r for r in lines('data/processed/complementary/inputs.jsonl') if r['id'] in ids}
    require(len(rows) == len(ids) and all(r['split'] == args.split for r in rows.values()), 'Scoring partition mismatch')
    labels = {r['id']: r['target'] for r in lines('data/processed/complementary/labels.jsonl') if r['id'] in ids}
    source = read('configs/complementary.json')['sources']['abcd']
    part = next(f for f in source['files'] if f['path'].endswith('abcd_v1.1.json.gz'))
    url = f"https://raw.githubusercontent.com/{source['repo']}/{source['revision']}/{part['path']}"
    raw = Path('data/raw/complementary/abcd', digest(url)[:20] + '.raw').read_bytes()
    require(digest(raw) == part['sha256'], 'Annotation source changed')
    schema = read('configs/abcd-schema-v2.json')
    procedures = {}
    for conversations in json.loads(gzip.decompress(raw)).values():
        for conversation in conversations:
            stage = 0
            for i, turn in enumerate(conversation['delexed']):
                key = f"abcd:{conversation['convo_id']}:{i}"
                if key in ids:
                    procedures[key] = (schema['procedureMap'][turn['targets'][0]], min(stage, 15))
                stage += int(turn['speaker'] == 'action')
    measurements, grouped = [], defaultdict(list)
    for record in saved:
        key = record.get('id', record.get('key'))
        prediction, label = record['prediction'], labels[key]
        values = dict(record['metrics'])
        values['toolDecisionAccuracy'] = float(
            prediction['action'] == label['action']
            and (label['action'] != 'call_tool' or prediction.get('tool') == label['tool'])
        )
        if label['action'] == 'call_tool':
            values['selectedToolRecall'] = float(prediction.get('tool') == label['tool'])
            if label.get('argumentsObservable'):
                values['observableToolExact'] = values['toolExact']
        if prediction['action'] == 'call_tool':
            values['toolCallPrecision'] = float(label['action'] == 'call_tool' and prediction.get('tool') == label.get('tool'))
            # A correct tool with hidden argument labels cannot be judged as a complete call.
            if label['action'] != 'call_tool' or label.get('argumentsObservable') or not values['toolCallPrecision']:
                values['observableCallPrecision'] = values.get('observableToolAndArgumentsExact', 0.0)
        if key in procedures and prediction.get('procedureIds'):
            procedure, stage = procedures[key]
            values.update(procedureTop1=float(prediction['procedureIds'][0] == procedure),
                          procedureTop3=float(procedure in prediction['procedureIds']),
                          progressStageAccuracy=float(prediction['progress']['stage'] == stage))
        if label['action'] == 'speak' and '?' in label.get('answer', ''):
            text = ' ' + canonical(label['answer']) + ' '
            fields = [name for name in schema['names'] if ' ' + canonical(name.replace('_', ' ')) + ' ' in text]
            if fields:
                selected = prediction.get('question', {}).get('field')
                values['explicitQuestionFieldAccuracy'] = float(selected in ['question:' + f for f in fields])
        measurements.append({'key': key, 'family': rows[key]['family'], 'metrics': values})
        for name, value in values.items():
            if value is not None:
                grouped[name].append(value)
    summary = {k: {'correctOrSum': sum(v), 'cases': len(v), 'mean': sum(v) / len(v)} for k, v in grouped.items()}
    output = Path(args.output)
    write(output, {'split': args.split, 'metrics': summary, 'predictionHash': digest(Path(args.predictions).read_bytes()),
                   'annotationHash': part['sha256'], 'scriptHash': digest(Path(__file__).read_bytes()),
                   'scope': 'Scoring only; procedure/progress truth never enters construction or provider inputs. Question field score covers only explicitly named fields in reference questions, not overall question necessity.'})
    jsonl(output.with_suffix('.jsonl'), measurements)
    print(summary)


if __name__ == '__main__':
    main()
