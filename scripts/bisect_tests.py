import subprocess,sys
# Collect ordered test nodes
p = subprocess.run([sys.executable,'-m','pytest','--collect-only','-q'], capture_output=True, text=True)
lines = [l.strip() for l in p.stdout.splitlines() if l.strip()]
files = []
seen = set()
for l in lines:
    if '::' in l:
        f = l.split('::',1)[0]
    else:
        f = l
    if f not in seen:
        seen.add(f)
        files.append(f)
print('Collected', len(files), 'files')
# target test
target='tests/test_rag.py::test_rag_agent_does_not_misroute_unrelated_query_with_marker_substring'
lo=0; hi=len(files)-1; culprit=None
while lo<=hi:
    mid=(lo+hi)//2
    subset=files[:mid+1]
    cmd=[sys.executable,'-m','pytest','-q']+subset+[target]
    print('Running subset up to index',mid, 'file', files[mid])
    r = subprocess.run(cmd, capture_output=True, text=True)
    out=r.stdout+r.stderr
    if 'FAILED' in out or 'FAIL' in out:
        culprit=mid
        hi=mid-1
        print('Failure observed with subset up to', files[mid])
    else:
        lo=mid+1
        print('No failure with subset up to', files[mid])
print('Culprit index:', culprit, 'file:', files[culprit] if culprit is not None else 'None')
