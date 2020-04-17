import urllib.request

def download_fasta(pid):
    url="https://www.uniprot.org/uniprot/?query="+pid+"&format=fasta&compress=no"
    try:
        req=urllib.request.urlopen(url)
        r = req.read()
        return r
    except urllib.error.HTTPError as e:
        print(pid,e.reason)
        return None

fp=open("protein.fa","wb")
filename="data.tsv"
for line in open(filename):
    arr=line.strip().split("\t")
    pid=arr[0]
    label=arr[1]
    print(pid,label)
    r=download_fasta(pid)
    fp.write((">"+pid+" "+label+"\n").encode('utf-8'))
    if r is not None:
        fp.write(r)
    else:
        fp.write((">"+pid+"\n\n").encode('utf-8'))
    fp.write(b"\n")

