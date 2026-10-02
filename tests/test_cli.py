from pathlib import Path
import json
import pytest
from ssr.cli import main, parser


def test_live_is_default_and_cached_option_is_removed():
    args = parser().parse_args(['--model','model','--image','image.jpg','--question','Question'])
    assert args.tools == 'live'
    with pytest.raises(SystemExit):
        parser().parse_args(['--model','model','--image','image.jpg','--tools','cached'])


def test_only_exhibition_example_and_dry_run(capsys):
    root=Path(__file__).resolve().parents[1]
    examples=list((root/'examples').glob('*'))
    examples=[p for p in examples if p.is_file()]
    assert [p.name for p in examples] == ['teaser-exhibition.jsonl']
    row=json.loads(examples[0].read_text())
    assert 'exhibition' in row['question'] and 'profound disruption' in row['question']
    main(['--model','Qwen/Qwen3-VL-4B-Instruct','--input-jsonl',str(examples[0]),'--dry-run'])
    data=json.loads(capsys.readouterr().out)
    assert data['tools']=='live' and data['examples']==1
