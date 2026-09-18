"""M5 fixed-path, byte-only source audit. No numerical CSV parsing or models.

Normalization is limited to leading UTF-8 BOM, CRLF -> LF and one terminal LF.
The embedded eight tests use only synthetic files, with exact unittest IDs.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

BASE=Path('/public/home/yueweiting/大论文')
EVIDENCE=BASE/'amd-execution-evidence/m5/m5-source-version-_rkt_o7f'
PAIRS={
    'Weather':(BASE/'AMD/data/weather.csv','weather.csv'),
    'ECL':(BASE/'AMD/data/electricity.csv','electricity.csv'),
    'Exchange':(BASE/'AMD/data/exchange_rate.csv','exchange_rate.csv'),
    'PJM':(BASE/'TimeXer/dataset/EPF/PJM.csv','PJM-git.csv'),
    'ETTh1':(BASE/'AMD/data/ETTh1.csv','ETTh1-author.csv'),
}
READS=[]


def snapshot(path):
    s=Path(path).stat()
    return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def scan(stream):
    raw_hash=hashlib.sha256();normal_hash=hashlib.sha256();pending=b''
    records=0;physical=0;quoted_newlines=0;blank=0;size=0;state='start';active=False
    columns=1;header_columns=None;column_mismatches=0;crlf=0;bom=False
    for raw in stream:
        raw_hash.update(raw);size+=len(raw);physical+=1
        line=raw
        if physical==1 and line.startswith(b'\xef\xbb\xbf'):line=line[3:];bom=True
        normalized=line.replace(b'\r\n',b'\n');combined=pending+normalized
        if combined:normal_hash.update(combined[:-1]);pending=combined[-1:]
        # Examine framing bytes only, never convert field values or dates.
        pos=0
        for m in re.finditer(b'[,"\r\n]',line):
            gap=m.start()-pos;ch=m.group();pos=m.end()
            if gap:
                if state=='closed':raise ValueError('bytes after closing quote')
                if state=='start':state='plain'
                active=True
            if ch==b'"':
                if state=='start':state='quoted';active=True
                elif state=='quoted':state='closed'
                elif state=='closed':state='quoted'  # escaped pair
                else:raise ValueError('quote inside unquoted field')
            elif ch==b',':
                if state!='quoted':state='start';columns+=1
                active=True
            elif ch==b'\r':
                if line[pos:pos+1]!=b'\n':raise ValueError('bare CR framing')
                crlf+=1
            elif ch==b'\n':
                if state=='quoted':quoted_newlines+=1
                else:
                    if active:
                        records+=1
                        if header_columns is None:header_columns=columns
                        elif columns!=header_columns:column_mismatches+=1
                    else:blank+=1
                    columns=1;state='start';active=False
        if pos<len(line):
            if state=='closed':raise ValueError('bytes after closing quote')
            if state=='start':state='plain'
            active=True
    if state=='quoted':raise ValueError('unterminated quoted field')
    if active:
        records+=1
        if header_columns is None:header_columns=columns
        elif columns!=header_columns:column_mismatches+=1
    if pending and pending!=b'\n':normal_hash.update(pending)
    return dict(sha256=raw_hash.hexdigest(),normalized_sha256=normal_hash.hexdigest(),bytes=size,
                records_including_header=records,physical_lines=physical,blank_records=blank,
                quoted_newlines=quoted_newlines,header_columns=header_columns,column_mismatches=column_mismatches,
                leading_bom=bom,crlf=crlf,terminal_lf=pending==b'\n',numerical_fields_decoded=0)


def measured(path):
    before=snapshot(path)
    with Path(path).open('rb') as f:result=scan(f)
    after=snapshot(path)
    READS.append(dict(path=str(path),purpose='raw SHA + normalized SHA + CSV byte framing',byte_range=[0,before['size']],before=before,after=after))
    if before!=after:raise RuntimeError('file changed while reading; comparison invalid')
    return dict(result,before=before,after=after)


def first_difference(a,b):
    offset=0;line=1
    with Path(a).open('rb') as x,Path(b).open('rb') as y:
        while True:
            xb=x.read(1024*1024);yb=y.read(1024*1024)
            if xb!=yb:
                limit=min(len(xb),len(yb));i=next((i for i in range(limit) if xb[i]!=yb[i]),limit)
                return dict(byte_offset=offset+i,local_physical_line=line+xb[:i].count(b'\n'))
            if not xb:return None
            offset+=len(xb);line+=xb.count(b'\n')


def compare(a,b):
    left=measured(a);right=measured(b)
    grade='A' if left['sha256']==right['sha256'] else 'B' if left['normalized_sha256']==right['normalized_sha256'] else 'C'
    difference=None
    if grade=='C':
        difference=first_difference(a,b)
        for p,meta in ((a,left),(b,right)):
            READS.append(dict(path=str(p),purpose='first byte difference only; no value output',byte_range=[0,min(meta['bytes'],((difference['byte_offset']//1048576)+1)*1048576)]))
    if snapshot(a)!=left['before'] or snapshot(b)!=right['before']:raise RuntimeError('file changed during pair comparison')
    return dict(local=left,reference=right,grade=grade,first_difference=difference,
                normalization=['leading UTF-8 BOM only','CRLF to LF only','remove one terminal LF only'])


def validate_pair(local,reference):
    expected={str(p.resolve()):str((EVIDENCE/'reference'/name).resolve()) for p,name in PAIRS.values()}
    if expected.get(str(Path(local).resolve()))!=str(Path(reference).resolve()):raise PermissionError('not an explicitly authorized file/object pair')


class AuditTests(unittest.TestCase):
    def test_identical_and_rows(self):
        r=scan(io.BytesIO(b'date,a,b\nq,1,2\nr,3,4'))
        self.assertEqual((r['records_including_header'],r['header_columns'],r['column_mismatches']),(3,3,0))
        self.assertFalse(r['terminal_lf'])
    def test_format_normalization(self):
        a=scan(io.BytesIO(b'\xef\xbb\xbfdate,a\r\nx,1\r\n'));b=scan(io.BytesIO(b'date,a\nx,1'))
        self.assertNotEqual(a['sha256'],b['sha256']);self.assertEqual(a['normalized_sha256'],b['normalized_sha256'])
    def test_content_difference_and_location(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b';a.write_bytes(b'h,x\na,1\n');b.write_bytes(b'h,x\na,2\n')
            c=compare(a,b);self.assertEqual(c['grade'],'C');self.assertEqual(c['first_difference'],{'byte_offset':6,'local_physical_line':2})
    def test_quoted_delimiters_and_escapes(self):
        r=scan(io.BytesIO(b'h,x\r\n"a,""b",2\r\n'))
        self.assertEqual((r['records_including_header'],r['quoted_newlines'],r['column_mismatches']),(2,0,0))
    def test_embedded_newline_flag(self):
        r=scan(io.BytesIO(b'h,x\n"a\nb",2\n'))
        self.assertEqual((r['records_including_header'],r['quoted_newlines']),(2,1))
    def test_invalid_structure(self):
        for data in (b'h\n"x',b'h\na"b\n',b'h\n"a"x\n',b'h\rx'):
            with self.subTest(data=data),self.assertRaises(ValueError):scan(io.BytesIO(data))
    def test_file_change_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.write_bytes(b'h\nx\n')
            with patch(__name__+'.snapshot',side_effect=[{'size':4},{'size':5}]):
                with self.assertRaises(RuntimeError):measured(p)
    def test_exact_path_binding(self):
        for p,name in PAIRS.values():validate_pair(p,EVIDENCE/'reference'/name)
        with self.assertRaises(PermissionError):validate_pair(BASE/'AMD/data/other.csv',EVIDENCE/'reference/weather.csv')


IDS=[f'__main__.AuditTests.{name}' for name in (
 'test_identical_and_rows','test_format_normalization','test_content_difference_and_location',
 'test_quoted_delimiters_and_escapes','test_embedded_newline_flag','test_invalid_structure',
 'test_file_change_refused','test_exact_path_binding')]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['selftest','compare']);args=parser.parse_args()
    if args.action=='selftest':
        (EVIDENCE/'exact-test-ids.json').write_text(json.dumps(IDS,indent=2)+'\n')
        result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(i) for i in IDS))
        (EVIDENCE/'synthetic-tests.json').write_text(json.dumps(dict(methods=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),success=result.wasSuccessful(),subcases={'invalid_structure':4,'exact_path_positive':5,'exact_path_negative':1}),indent=2)+'\n')
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if not json.loads((EVIDENCE/'synthetic-tests.json').read_text())['success']:raise RuntimeError('synthetic acceptance required first')
    results={}
    try:
        for name,(p,f) in PAIRS.items():
            ref=EVIDENCE/'reference'/f;validate_pair(p,ref)
            if not ref.exists():results[name]={'grade':'D','reason':'reference unavailable'};continue
            results[name]=compare(p,ref)
            if name=='PJM':
                x=results[name]['local']
                if x['quoted_newlines'] or x['blank_records'] or x['column_mismatches']:raise ValueError('PJM framing needs review; no approximate n')
                n=x['records_including_header']-1;train=7*n//10;val_end=n-2*n//10
                w={'train':train-168-24+1,'validation':val_end-train-24+1,'test':n-val_end-24+1}
                results[name]['arithmetic_only']=dict(n=n,endpoints=[train,val_end,n],windows=w,batch=128,
                    batches={k:(v//128 if k=='train' else (v+127)//128) for k,v in w.items()},remainders={k:v%128 for k,v in w.items()},test_values_not_parsed=True)
    finally:
        (EVIDENCE/'comparison-results.json').write_text(json.dumps(results,indent=2)+'\n')
        (EVIDENCE/'byte-access-ledger.json').write_text(json.dumps(READS,indent=2)+'\n')


if __name__=='__main__':main()
